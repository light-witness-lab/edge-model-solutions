package verify

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import dev.streetsnap.app.edge.ByteBpeTokenizer
import dev.streetsnap.app.edge.D1Decision
import dev.streetsnap.app.edge.D1Prompt
import java.awt.BasicStroke
import java.awt.Color
import java.awt.Font
import java.awt.Graphics2D
import java.awt.RenderingHints
import java.awt.geom.Ellipse2D
import java.awt.geom.Rectangle2D
import java.awt.image.BufferedImage
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import org.json.JSONArray
import org.json.JSONObject

/**
 * T36 input-resolution probe for the d1-omni-600M vision graph (desktop ORT 1.22).
 *
 * NaFlex/SigLIP2 feeds a variable-length patch tensor with `spatial_shapes`, so the
 * same graph ought to accept coarser grids: 512 -> 32x32 = 1024 patches (256 prefix),
 * 384 -> 24x24 = 576 (144 prefix), 320 -> 20x20 = 400 (100 prefix). This probe checks
 * (a) whether the *exported* graph really accepts those shapes, (b) vision latency at
 * each grid, and (c) whether shrinking the input changes the v0 (shipped) choice on the
 * frozen 20-sheet bias/discrimination set. It is purely exploratory for a possible
 * on-device speed-up; it does NOT change any app contract (that would be a separate,
 * app+manifest task). Loaded-only evidence regardless of outcome.
 *
 * The same programmatic content is rasterised independently at each side length (all
 * geometry scaled by side/512), so there is no downscale-resampling confound and the
 * only difference between arms is the patch grid.
 *
 * Pre-registered decision rule (written before any number was produced):
 *  R1 latency:  recommend considering 320 only if its median per-sheet vision latency is
 *              <= 0.70x the 512 arm's (>=30% faster) on the same JVM run;
 *  R2 accuracy: relative to the 512 arm, neither 384 nor 320 may lose more than 2/18
 *              argmax-correct sheets, nor more than 2/18 gate-selected-correct sheets;
 *  R3 bias:     P1/P2 mean max-prob must not rise by more than 0.05, and the most-voted
 *              physical panel on the 18 discrimination sheets must not gain votes;
 *  R4 shape:    if a lower resolution fails to run on the exported graph, report
 *              "graph-fixed-at-512" (re-export required) — that is itself the result.
 */

private const val MODEL_PIN = "onnx-community/d1-omni-600M-ONNX@4ebc1b9 (vision_encoder_fp16 + decision_q4)"
private const val MIN_MARGIN = 0.10
private const val PATCH = 16
private const val PATCH_DIM = 3 * PATCH * PATCH // 768
private val SIDES = listOf(512, 384, 320)

// Shipped wording, inlined exactly as in the frozen prompt probe (comp-choice-v0).
private const val V0_STATE = ""
private const val V0_QUESTION = "Which crop gives the best composition?"
private val V0_OPTIONS = listOf("a: crop option A", "b: crop option B", "c: crop option C")

internal data class SheetProbe(val id: String, val kind: String, val goodPanel: Int?)

/** Rasterises one probe independently at [side]; [f] scales every 512-based coordinate. */
internal class ResolutionSheets(private val side: Int) {
    private val f = side / 512f
    private val ground = Color(244, 244, 244)
    private val border = Color(210, 210, 210)
    private val photoBg = Color(176, 176, 176)
    private val photoBgLow = Color(150, 150, 150)

    private val photoRects = listOf(
        Rectangle2D.Float(8f * f, 40f * f, 240f * f, 208f * f),
        Rectangle2D.Float(264f * f, 40f * f, 240f * f, 208f * f),
        Rectangle2D.Float(8f * f, 296f * f, 240f * f, 208f * f),
    )

    private fun paintScene(
        g: Graphics2D, r: Rectangle2D.Float,
        subjectColor: Color, round: Boolean, horizon: Boolean,
        cx: Float, cy: Float, radius: Float,
    ) {
        if (horizon) {
            g.color = photoBgLow
            g.fill(Rectangle2D.Float(r.x, r.y + r.height * 0.62f, r.width, r.height * 0.38f))
            g.color = Color(120, 120, 120)
            g.drawLine(r.x.toInt(), (r.y + r.height * 0.62f).toInt(),
                (r.x + r.width).toInt(), (r.y + r.height * 0.62f).toInt())
        }
        g.color = subjectColor
        val shape = if (round) Ellipse2D.Float(cx - radius, cy - radius, radius * 2, radius * 2)
        else Rectangle2D.Float(cx - radius, cy - radius, radius * 2, radius * 2)
        g.fill(shape)
    }

    private fun goodContent(scene: Int): (Graphics2D, Rectangle2D.Float) -> Unit {
        val thirdsX = floatArrayOf(1f / 3f, 2f / 3f, 1f / 3f, 2f / 3f)
        val thirdsY = floatArrayOf(1f / 3f, 1f / 3f, 2f / 3f, 2f / 3f)
        val tx = thirdsX[scene % 4]; val ty = thirdsY[scene % 4]
        val subject = if (scene % 2 == 0) Color(232, 214, 140) else Color(58, 60, 74)
        val round = scene % 2 == 0
        val horizon = scene % 3 != 0
        return { g, r -> paintScene(g, r, subject, round, horizon,
            r.x + r.width * tx, r.y + r.height * ty, 34f * f) }
    }

    private fun cutContent(scene: Int): (Graphics2D, Rectangle2D.Float) -> Unit {
        val subject = if (scene % 2 == 0) Color(232, 214, 140) else Color(58, 60, 74)
        val round = scene % 2 == 1
        val cutRight = scene % 2 == 0
        return { g, r ->
            paintScene(g, r, subject, round, horizon = true,
                if (cutRight) r.x + r.width else r.x, r.y + r.height * 0.5f, 40f * f)
        }
    }

    private fun cornerContent(scene: Int): (Graphics2D, Rectangle2D.Float) -> Unit {
        val subject = if (scene % 3 == 0) Color(232, 214, 140) else Color(58, 60, 74)
        val corners = arrayOf(
            floatArrayOf(0f, 0f), floatArrayOf(1f, 0f), floatArrayOf(0f, 1f), floatArrayOf(1f, 1f),
        )
        val c = corners[scene % 4]
        return { g, r ->
            paintScene(g, r, subject, round = true, horizon = false,
                r.x + r.width * c[0], r.y + r.height * c[1], 52f * f)
        }
    }

    fun raster(probe: SheetProbe): BufferedImage {
        val img = BufferedImage(side, side, BufferedImage.TYPE_INT_RGB)
        val g = img.createGraphics()
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
        g.color = ground
        g.fillRect(0, 0, side, side)
        g.font = Font(Font.SANS_SERIF, Font.BOLD, (30 * f).toInt().coerceAtLeast(8))
        g.color = Color.BLACK
        listOf("A" to 128f * f, "B" to 384f * f, "C" to 128f * f).forEachIndexed { i, (letter, cx) ->
            val y = if (i < 2) 0f else 256f * f
            val wm = g.fontMetrics.stringWidth(letter)
            g.drawString(letter, (cx - wm / 2f), y + 31f * f)
        }
        // Paint each photo panel: background + border, then probe-specific content.
        val contents = when {
            probe.kind == "bias" && probe.id.startsWith("P1") -> List(3) { { _: Graphics2D, _: Rectangle2D.Float -> } }
            probe.kind == "bias" -> List(3) { goodContent(0) }
            else -> {
                val good = probe.goodPanel!!
                var badKinds = 0
                List(3) { panel ->
                    if (panel == good) goodContent(sceneOf(probe.id))
                    else if (badKinds++ % 2 == 0) cutContent(sceneOf(probe.id))
                    else cornerContent(sceneOf(probe.id))
                }
            }
        }
        photoRects.forEachIndexed { panel, r ->
            g.color = photoBg; g.fill(r)
            contents[panel](g, r)
            g.color = border; g.stroke = BasicStroke(maxOf(1f, f)); g.draw(r)
        }
        g.dispose()
        return img
    }

    private fun sceneOf(id: String): Int =
        Regex("D-s(\\d)-").find(id)?.groupValues?.get(1)?.toInt() ?: 0
}

private fun buildProbes(): List<SheetProbe> {
    val probes = ArrayList<SheetProbe>()
    probes += SheetProbe("P1-uniform", "bias", null)
    probes += SheetProbe("P2-same-scene", "bias", null)
    for (scene in 0..5) for (goodPanel in 0..2) {
        probes += SheetProbe("D-s${scene}-good${'A' + goodPanel}", "discrimination", goodPanel)
    }
    return probes
}

/** Variable-grid patch layout; the counterpart of the frozen 512-only D1NaFlex snapshot. */
private fun toPatches(argb: IntArray, side: Int): FloatArray {
    require(side % PATCH == 0) { "side must be a multiple of $PATCH, got $side" }
    require(argb.size == side * side)
    val grid = side / PATCH
    val out = FloatArray(grid * grid * PATCH_DIM)
    for (py in 0 until grid) for (px in 0 until grid) {
        val patchRow = (py * grid + px) * PATCH_DIM
        for (dy in 0 until PATCH) for (dx in 0 until PATCH) {
            val color = argb[(py * PATCH + dy) * side + (px * PATCH + dx)]
            val within = (dy * PATCH + dx) * 3
            out[patchRow + within] = ((color ushr 16) and 255) / 127.5f - 1f
            out[patchRow + within + 1] = ((color ushr 8) and 255) / 127.5f - 1f
            out[patchRow + within + 2] = (color and 255) / 127.5f - 1f
        }
    }
    return out
}

internal class ResolutionEngine(modelDir: File) {
    private val env = OrtEnvironment.getEnvironment()
    private val config = JSONObject(File(modelDir, "config.json").readText())
    val hiddenSize: Int = config.getJSONObject("text_config").getInt("hidden_size")
    private val imageTextLength = config.optInt("image_text_length", 896)
    private val bosTokenId = config.getInt("bos_token_id")
    private val vision: OrtSession
    private val embeddings: OrtSession
    private val decision: OrtSession
    private val tokenizer = ByteBpeTokenizer(File(modelDir, "tokenizer.json"))

    init {
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        vision = env.createSession(File(modelDir, "onnx/vision_encoder_fp16.onnx").absolutePath, options)
        embeddings = env.createSession(File(modelDir, "onnx/embed_tokens_quantized.onnx").absolutePath, options)
        decision = env.createSession(File(modelDir, "onnx/decision_q4.onnx").absolutePath, options)
    }

    /** @return vision features (length implies the prefix embeddings) or throws if the graph rejects this grid. */
    fun visionFeatures(argb: IntArray, side: Int): FloatArray {
        val grid = side / PATCH
        val patches = grid * grid
        val pixels = toPatches(argb, side)
        OnnxTensor.createTensor(env, FloatBuffer.wrap(pixels),
            longArrayOf(1, patches.toLong(), PATCH_DIM.toLong())).use { pt ->
            OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(patches) { 1L }),
                longArrayOf(1, patches.toLong())).use { mask ->
                OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(grid.toLong(), grid.toLong())),
                    longArrayOf(1, 2)).use { shapes ->
                        vision.run(mapOf("pixel_values" to pt, "pixel_attention_mask" to mask,
                            "spatial_shapes" to shapes)).use { out ->
                            val tensor = out.get("image_features").get() as OnnxTensor
                            return FloatArray(tensor.floatBuffer.remaining()).also { tensor.floatBuffer.get(it) }
                        }
                    }
                }
            }
    }

    fun decide(features: FloatArray): JSONObject {
        val encoded = D1Prompt.encode(
            encode = { text -> tokenizer.encode(text) },
            idOf = { token -> if (token == "<bos>") bosTokenId else tokenizer.specialId(token) },
            state = V0_STATE, instructions = V0_QUESTION, options = V0_OPTIONS,
            maxLen = imageTextLength,
        )
        val textEmbeddings: FloatArray
        OnnxTensor.createTensor(env, LongBuffer.wrap(encoded.ids),
            longArrayOf(1, encoded.ids.size.toLong())).use { tokenTensor ->
            embeddings.run(mapOf("input_ids" to tokenTensor)).use { out ->
                val tensor = out.get("inputs_embeds").get() as OnnxTensor
                textEmbeddings = FloatArray(tensor.floatBuffer.remaining()).also { tensor.floatBuffer.get(it) }
            }
        }
        val prefixLength = features.size / hiddenSize
        val combined = FloatArray(prefixLength * hiddenSize + textEmbeddings.size)
        features.copyInto(combined, 0)
        textEmbeddings.copyInto(combined, prefixLength * hiddenSize)
        val totalLength = prefixLength + encoded.ids.size
        val logits: FloatArray
        val tensors = listOf(
            OnnxTensor.createTensor(env, FloatBuffer.wrap(combined),
                longArrayOf(1, totalLength.toLong(), hiddenSize.toLong())),
            OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(totalLength) { 1L }),
                longArrayOf(1, totalLength.toLong())),
            OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(prefixLength.toLong())), longArrayOf(1)),
            OnnxTensor.createTensor(env, LongBuffer.wrap(encoded.markerPositions),
                longArrayOf(1, encoded.optionCount.toLong())),
            OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(encoded.optionCount) { 1L }),
                longArrayOf(1, encoded.optionCount.toLong())),
            OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(D1Prompt.QTYPE_CHOICE)), longArrayOf(1)),
        )
        try {
            val names = listOf("inputs_embeds", "attention_mask", "prefix_len",
                "marker_pos", "marker_mask", "qtype")
            decision.run(names.zip(tensors).toMap(), setOf("logits")).use { out ->
                val tensor = out.get("logits").get() as OnnxTensor
                logits = FloatArray(tensor.floatBuffer.remaining()).also { tensor.floatBuffer.get(it) }
            }
        } finally {
            tensors.asReversed().forEach { it.close() }
        }
        val ranking = D1Decision.toRanking(logits)
        val probs = ranking?.probabilities?.toList().orEmpty()
        return JSONObject()
            .put("probs", JSONArray(probs.map { Math.round(it * 10000.0) / 10000.0 }))
            .put("selected", ranking?.selectedIndex ?: JSONObject.NULL)
            .put("margin", ranking?.margin?.let { Math.round(it * 10000.0) / 10000.0 } ?: JSONObject.NULL)
            .put("gate_pass", (ranking?.margin ?: 0f) >= MIN_MARGIN)
    }

    fun close() = listOf(vision, embeddings, decision).forEach { runCatching { it.close() } }
}

private fun median(values: List<Long>): Long = values.sorted().let { it[it.size / 2] }

fun main(args: Array<String>) {
    val modelDir = File(args.firstOrNull() ?: error("usage: ResolutionProbe <d1-model-dir> [out.json]"))
    val outFile = File(args.getOrNull(1) ?: "resolution-eval.json")
    println("model dir: $modelDir")

    val probes = buildProbes()
    val engine = ResolutionEngine(modelDir)
    val armsJson = JSONArray()
    try {
        for (side in SIDES) {
            val grid = side / PATCH
            val arm = JSONObject()
                .put("side", side).put("grid", grid)
                .put("patches", grid * grid).put("expected_prefix", grid * grid / 4)
            val sheets = ResolutionSheets(side)
            try {
                // Warm up this shape (graph optimisation/first-run allocations excluded).
                repeat(2) { engine.visionFeatures(sheets.raster(probes.first()).getRGB(0, 0, side, side, null, 0, side), side) }

                // Stable latency: vision only, one fixed sheet repeated 11 times.
                val fixedArgb = sheets.raster(probes.first { it.kind == "discrimination" })
                    .getRGB(0, 0, side, side, null, 0, side)
                val repeatedMs = (1..11).map {
                    val t0 = System.nanoTime(); engine.visionFeatures(fixedArgb, side); (System.nanoTime() - t0) / 1_000_000
                }

                // Per-sheet: vision latency + v0 decision on every probe.
                val rows = JSONArray()
                val perSheetVisionMs = ArrayList<Long>()
                var actualPrefix = -1
                for (probe in probes) {
                    val argb = sheets.raster(probe).getRGB(0, 0, side, side, null, 0, side)
                    val t0 = System.nanoTime()
                    val features = engine.visionFeatures(argb, side)
                    val visionMs = (System.nanoTime() - t0) / 1_000_000
                    perSheetVisionMs += visionMs
                    if (actualPrefix < 0) actualPrefix = features.size / engine.hiddenSize
                    val decisionJson = engine.decide(features)
                    val selected = if (decisionJson.isNull("selected")) -1 else decisionJson.getInt("selected")
                    val correct = probe.goodPanel != null && selected == probe.goodPanel
                    rows.put(JSONObject()
                        .put("id", probe.id).put("kind", probe.kind)
                        .put("good_panel", probe.goodPanel?.let { "${'A' + it}" } ?: JSONObject.NULL)
                        .put("vision_ms", visionMs)
                        .put("probs", decisionJson.getJSONArray("probs"))
                        .put("selected", decisionJson.get("selected"))
                        .put("gate_pass", decisionJson.getBoolean("gate_pass"))
                        .put("correct", correct))
                }
                arm.put("status", "ok").put("actual_prefix", actualPrefix)
                arm.put("sheets", rows)
                arm.put("vision_ms", JSONObject()
                    .put("per_sheet_median", median(perSheetVisionMs))
                    .put("per_sheet_mean", Math.round(perSheetVisionMs.average()))
                    .put("repeat11_median", median(repeatedMs))
                    .put("repeat11_min", repeatedMs.min()))
                armsJson.put(arm)
                println("side=$side ok prefix=${"see sheets"} repeat11_median=${median(repeatedMs)}ms")
            } catch (error: Exception) {
                arm.put("status", "failed").put("error", error.javaClass.simpleName + ": " + error.message)
                armsJson.put(arm)
                println("side=$side FAILED: ${error.javaClass.simpleName}: ${error.message}")
            }
        }
    } finally {
        engine.close()
    }

    val report = JSONObject()
        .put("purpose", "T36 d1 input-resolution probe (512/384/320): vision latency + v0 choice stability on the frozen 20-sheet set; exploratory for on-device speed, not aesthetic truth")
        .put("model", MODEL_PIN)
        .put("runtime", "onnxruntime 1.22.0, desktop JVM, 4 threads, CPU; deterministic sheets independently rasterised at each side")
        .put("date", "2026-10-19")
        .put("prompt", "v0 shipped wording only (comp-choice-v1 was rejected 2026-10-18)")
        .put("preregistered_rule", JSONObject()
            .put("R1_latency", "consider 320 only if repeat11_median(320) <= 0.70 * repeat11_median(512)")
            .put("R2_accuracy", "vs 512, neither 384 nor 320 loses >2/18 argmax-correct nor >2/18 gate-selected-correct")
            .put("R3_bias", "P1/P2 mean max-prob rise <= 0.05 and most-voted physical panel does not gain votes")
            .put("R4_shape", "a shape failure at lower resolution => graph-fixed-at-512 (re-export required)"))
        .put("arms", armsJson)

    // Verdict computed from the numbers, not hand-picked.
    val ok = (0 until armsJson.length()).map { armsJson.getJSONObject(it) }.filter { it.optString("status") == "ok" }
    val bySide = ok.associateBy { it.getInt("side") }
    val verdict = JSONObject()
    val base = bySide[512]
    if (base == null) {
        verdict.put("recommendation", "inconclusive").put("reason", "512 arm did not run")
    } else {
        fun summary(arm: JSONObject): JSONObject {
            val rows = arm.getJSONArray("sheets")
            val list = (0 until rows.length()).map { rows.getJSONObject(it) }
            val disc = list.filter { it.getString("kind") == "discrimination" }
            val bias = list.filter { it.getString("kind") == "bias" }
            val argmaxCorrect = disc.count { it.getBoolean("correct") }
            val gateCorrect = disc.count {
                it.getBoolean("correct") && it.getBoolean("gate_pass")
            }
            val biasMax = bias.map {
                it.getJSONArray("probs").toList().filterIsInstance<Double>().max()
            }
            val physicalVotes = IntArray(3)
            for (row in disc) if (!row.isNull("selected")) physicalVotes[row.getInt("selected")]++
            return JSONObject()
                .put("argmax_correct", argmaxCorrect).put("gate_selected_correct", gateCorrect)
                .put("bias_mean_max_prob", Math.round(biasMax.average() * 10000.0) / 10000.0)
                .put("physical_argmax_votes", JSONArray(physicalVotes.toList()))
        }
        val s512 = summary(base)
        verdict.put("summary_512", s512)
        val latency512 = base.getJSONObject("vision_ms").getLong("repeat11_median")
        for (side in listOf(384, 320)) {
            val arm = bySide[side] ?: run {
                verdict.put("side_$side", JSONObject().put("runnable", false)); continue
            }
            val s = summary(arm)
            verdict.put("summary_$side", s)
            val lat = arm.getJSONObject("vision_ms").getLong("repeat11_median")
            val speedRatio = Math.round(lat * 10000.0 / latency512) / 10000.0
            val r1 = lat <= (latency512 * 0.70).toLong()
            val r2arg = s512.getInt("argmax_correct") - s.getInt("argmax_correct") <= 2
            val r2gate = s512.getInt("gate_selected_correct") - s.getInt("gate_selected_correct") <= 2
            val r3bias = s.getDouble("bias_mean_max_prob") - s512.getDouble("bias_mean_max_prob") <= 0.05
            val v512 = s512.getJSONArray("physical_argmax_votes").toList().filterIsInstance<Int>().max()
            val vs = s.getJSONArray("physical_argmax_votes").toList().filterIsInstance<Int>().max()
            val r3panel = vs <= v512
            verdict.put("side_$side", JSONObject()
                .put("runnable", true)
                .put("vision_latency_ratio_vs_512", speedRatio)
                .put("R1_latency_pass", r1)
                .put("R2_argmax_pass", r2arg).put("R2_gate_pass", r2gate)
                .put("R3_bias_pass", r3bias).put("R3_panel_pass", r3panel)
                .put("recommend_consider", side == 320 && r1 && r2arg && r2gate && r3bias && r3panel))
        }
        val s320 = verdict.optJSONObject("side_320")
        verdict.put("recommendation", when {
            s320?.optBoolean("runnable") == false -> "graph-fixed-at-512-reexport-required"
            s320?.optBoolean("recommend_consider") == true -> "consider-320-for-future-app-contract-task"
            else -> "no-change-loaded-only"
        })
    }
    report.put("verdict", verdict)
    outFile.writeText(report.toString(2))
    println("\n== verdict: ${verdict.optString("recommendation")}")
    println("wrote $outFile")
}

package verify

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import dev.streetsnap.app.edge.ByteBpeTokenizer
import dev.streetsnap.app.edge.D1Decision
import dev.streetsnap.app.edge.D1NaFlex
import dev.streetsnap.app.edge.D1Prompt
import java.awt.BasicStroke
import java.awt.Color
import java.awt.Font
import java.awt.Graphics2D
import java.awt.RenderingHints
import java.awt.geom.Line2D
import java.awt.geom.Rectangle2D
import java.awt.image.BufferedImage
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import org.json.JSONArray
import org.json.JSONObject

/**
 * Isolated candidate-diversity evaluation for the composition choice task (T10).
 *
 * Context: T07 showed stronger prompt wording does not move d1 on synthetic sheets
 * (see evaluations/d1-prompt-v1-eval-2026-10-18.json), so T10 fuses richer composition
 * grammar on the CANDIDATE side: when strong scene evidence (symmetry / vertical lines /
 * horizon / rhythm / tonal contrast / convergence) has no geometrically matching crop in
 * the energy-ranked top three, a full-frame style-seed crop replaces the third candidate.
 * The model therefore sees three crops of ONE frame that carry genuinely different
 * composition grammars, instead of three near-duplicates. This probe measures whether the
 * model can tell the grammar-correct crop from the wrong ones in that exact setting.
 *
 * The probe is programmatic (fixed, no RNG, never used to tune the rule layer). Each of
 * the six styles has one 416x416 source "frame"; the three panels are three square crops
 * of that same frame rendered with the app's real CandidateContactSheet geometry (a
 * 208x208 square centred in a 240x208 photo area). GOOD keeps the style's structure;
 * BAD-1/BAD-2 crop into a corner/sliver or cut the structure away. GOOD rotates through
 * A/B/C, so following content across physical positions is the permutation check.
 *
 * Pre-registered decision rule (set before seeing any number):
 *  1. content following: argmax correct >= 12/18 (2/3) -> the model reads style grammar;
 *     a result near the T07 v0 level (9/18) with strong P1/P2 max-prob (>= 0.40) is a
 *     positional responder, and style seeding is only a safe diversity change (the model
 *     must stay a weak signal; the 0.10 gate is not relaxed);
 *  2. gate-selected-correct (margin >= 0.10) is reported but never used to claim aesthetic
 *     truth; geometric shapes are far from real photographs;
 *  3. fewer than 4/18 between any two style groups = no observable difference.
 * Loaded-only evidence stays loaded-only regardless of the outcome.
 */

private const val MODEL_PIN = "onnx-community/d1-omni-600M-ONNX@4ebc1b9 (decision_q4)"
private const val MIN_MARGIN = 0.10
private const val FRAME = 416

// Exact shipped wording, inlined so the experiment is reproducible if the app changes.
private const val V0_STATE = ""
private const val V0_QUESTION = "Which crop gives the best composition?"
private val V0_OPTIONS = listOf("a: crop option A", "b: crop option B", "c: crop option C")

/** Normalized square crop: left, top, side length — same contract as the app's box. */
private data class Crop(val left: Float, val top: Float, val side: Float) {
    val right get() = left + side
    val bottom get() = top + side
}

private enum class Style { SYMMETRY, HORIZON, REPETITION, VERTICAL, CONVERGENCE, CONTRAST }

private data class Probe(
    val id: String,
    val style: Style?,
    val kind: String,
    val goodPanel: Int?,
    val image: BufferedImage,
)

private class StyleSheets {
    private val ground = Color(244, 244, 244)
    private val border = Color(210, 210, 210)

    private val photoAreas = listOf(
        Rectangle2D.Float(8f, 40f, 240f, 208f),
        Rectangle2D.Float(264f, 40f, 240f, 208f),
        Rectangle2D.Float(8f, 296f, 240f, 208f),
    )

    // ---- The six synthetic "camera frames" -------------------------------------

    private fun newFrame(): BufferedImage =
        BufferedImage(FRAME, FRAME, BufferedImage.TYPE_INT_RGB)

    private fun mirror(g: Graphics2D, block: (Graphics2D) -> Unit) {
        // Draw through a translation so a rect at x on the left is mirrored to FRAME-x.
        val saved = g.transform
        block(g)
        g.translate(FRAME.toDouble(), 0.0)
        g.scale(-1.0, 1.0)
        block(g)
        g.transform = saved
    }

    private fun frameFor(style: Style): Pair<BufferedImage, Triple<Crop, Crop, Crop>> {
        val img = newFrame()
        val g = img.createGraphics()
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
        g.setRenderingHint(RenderingHints.KEY_STROKE_CONTROL, RenderingHints.VALUE_STROKE_PURE)
        lateinit var crops: Triple<Crop, Crop, Crop>
        when (style) {
            Style.SYMMETRY -> {
                g.color = Color(150, 150, 150)
                g.fillRect(0, 0, FRAME, FRAME)
                // Mirrored facade: whatever is drawn on the left is mirrored exactly.
                mirror(g) { gg ->
                    gg.color = Color(96, 96, 96)
                    gg.fillRect(28, 70, 60, 46)
                    gg.fillRect(36, 168, 60, 46)
                    gg.fillRect(30, 268, 60, 46)
                    gg.color = Color(190, 190, 190)
                    gg.fillRect(24, 0, 12, FRAME)
                }
                g.color = Color(64, 64, 64)
                g.fillRect(FRAME / 2 - 24, 232, 48, FRAME - 232)
                g.color = Color(110, 110, 110)
                g.stroke = BasicStroke(3f)
                g.drawLine(0, 140, FRAME, 140)
                g.drawLine(0, 240, FRAME, 240)
                crops = Triple(
                    Crop(0.02f, 0.02f, 0.96f),
                    Crop(0.05f, 0.30f, 0.34f),
                    Crop(0.42f, 0.10f, 0.56f),
                )
            }
            Style.HORIZON -> {
                g.color = Color(216, 216, 216)
                g.fillRect(0, 0, FRAME, 137)
                g.color = Color(96, 96, 96)
                g.fillRect(0, 137, FRAME, FRAME - 137)
                g.color = Color(70, 70, 70)
                g.stroke = BasicStroke(4f)
                g.drawLine(0, 137, FRAME, 137)
                g.color = Color(120, 120, 120)
                g.stroke = BasicStroke(3f)
                g.drawLine(60, 250, 200, 250)
                g.drawLine(250, 320, 380, 320)
                crops = Triple(
                    Crop(0.0f, 0.0f, 1.0f),
                    Crop(0.0f, 0.0f, 0.46f),
                    Crop(0.20f, 0.54f, 0.46f),
                )
            }
            Style.REPETITION -> {
                g.color = Color(44, 44, 44)
                g.fillRect(0, 0, FRAME, FRAME)
                g.color = Color(210, 210, 210)
                for (i in 0..7) g.fillRect(24 + i * 48, 24, 18, FRAME - 48)
                crops = Triple(
                    Crop(0.02f, 0.02f, 0.96f),
                    Crop(0.02f, 0.10f, 0.32f),
                    Crop(0.50f, 0.05f, 0.50f),
                )
            }
            Style.VERTICAL -> {
                g.color = Color(196, 196, 196)
                g.fillRect(0, 0, FRAME, FRAME)
                g.color = Color(58, 58, 58)
                intArrayOf(40, 150, 266, 376).forEach { x -> g.fillRect(x, 12, 14, FRAME - 24) }
                crops = Triple(
                    Crop(0.02f, 0.02f, 0.96f),
                    Crop(0.36f, 0.05f, 0.28f),
                    Crop(0.42f, 0.05f, 0.56f),
                )
            }
            Style.CONVERGENCE -> {
                g.color = Color(38, 38, 38)
                g.fillRect(0, 0, FRAME, FRAME)
                g.color = Color(214, 214, 214)
                g.stroke = BasicStroke(9f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND)
                g.draw(Line2D.Float(0f, FRAME.toFloat(), FRAME / 2f, 60f))
                g.draw(Line2D.Float(FRAME.toFloat(), FRAME.toFloat(), FRAME / 2f, 60f))
                g.color = Color(232, 214, 140)
                g.fillOval(FRAME / 2 - 13, 40, 26, 40)
                crops = Triple(
                    Crop(0.02f, 0.0f, 0.96f),
                    Crop(0.10f, 0.50f, 0.50f),
                    Crop(0.50f, 0.10f, 0.50f),
                )
            }
            Style.CONTRAST -> {
                g.color = Color(36, 36, 36)
                g.fillRect(0, 0, FRAME / 2, FRAME)
                g.color = Color(230, 230, 230)
                g.fillRect(FRAME / 2, 0, FRAME / 2, FRAME)
                g.color = Color(150, 120, 60)
                g.fillOval(FRAME / 2 - 44, 168, 88, 88)
                crops = Triple(
                    Crop(0.02f, 0.02f, 0.96f),
                    Crop(0.0f, 0.10f, 0.46f),
                    Crop(0.54f, 0.10f, 0.46f),
                )
            }
        }
        g.dispose()
        return img to crops
    }

    // ---- Contact sheet, mirroring CandidateContactSheet -------------------------

    private fun newSheet(): BufferedImage {
        val img = BufferedImage(512, 512, BufferedImage.TYPE_INT_RGB)
        val g = img.createGraphics()
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
        g.color = ground
        g.fillRect(0, 0, 512, 512)
        g.font = Font(Font.SANS_SERIF, Font.BOLD, 30)
        g.color = Color.BLACK
        listOf("A" to 128f, "B" to 384f, "C" to 128f).forEachIndexed { i, (letter, cx) ->
            val y = if (i < 2) 0f else 256f
            val w = g.fontMetrics.stringWidth(letter)
            g.drawString(letter, cx - w / 2f, y + 31f)
        }
        g.dispose()
        return img
    }

    private fun drawCrop(sheet: BufferedImage, panel: Int, frame: BufferedImage, crop: Crop) {
        val area = photoAreas[panel]
        val src = Rectangle2D.Float(
            crop.left * FRAME, crop.top * FRAME, crop.side * FRAME, crop.side * FRAME,
        )
        // Square crop contained in 240x208 -> 208x208 centred, exactly like the app.
        val dest = Rectangle2D.Float(area.centerX.toInt() - 104f, area.y, 208f, 208f)
        val g = sheet.createGraphics()
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR)
        g.drawImage(frame,
            dest.x.toInt(), dest.y.toInt(),
            (dest.x + dest.width).toInt(), (dest.y + dest.height).toInt(),
            src.x.toInt(), src.y.toInt(),
            (src.x + src.width).toInt(), (src.y + src.height).toInt(), null)
        g.color = border
        g.stroke = BasicStroke(1f)
        g.draw(dest)
        g.dispose()
    }

    fun buildProbes(): List<Probe> {
        val probes = ArrayList<Probe>()

        // Bias probes: identical panels carry no information to distinguish.
        val anyFrame = frameFor(Style.SYMMETRY)
        probes += Probe("P1-uniform", null, "bias", null, newSheet().also { sheet ->
            for (p in 0..2) drawCrop(sheet, p, anyFrame.first, Crop(0.02f, 0.02f, 0.96f))
        })
        probes += Probe("P2-same-crop", null, "bias", null, newSheet().also { sheet ->
            for (p in 0..2) drawCrop(sheet, p, anyFrame.first, anyFrame.second.first)
        })

        // Discrimination: GOOD crop rotates A/B/C across the six style frames.
        for (style in Style.entries) {
            val (frame, crops) = frameFor(style)
            val ordered = listOf(crops.first, crops.second, crops.third)
            for (goodPanel in 0..2) {
                val sheet = newSheet()
                var bad = 0
                for (panel in 0..2) {
                    val crop = if (panel == goodPanel) ordered[0] else ordered[1 + (bad++ % 2)]
                    drawCrop(sheet, panel, frame, crop)
                }
                probes += Probe("D-${style.name}-good${'A' + goodPanel}", style,
                    "discrimination", goodPanel, sheet)
            }
        }
        return probes
    }
}

private class Engine(modelDir: File) {
    private val env = OrtEnvironment.getEnvironment()
    private val config = JSONObject(File(modelDir, "config.json").readText())
    private val hiddenSize = config.getJSONObject("text_config").getInt("hidden_size")
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

    fun visionFeatures(argb: IntArray): FloatArray {
        val patches = D1NaFlex.toPatchPixels(argb)
        OnnxTensor.createTensor(env, FloatBuffer.wrap(patches),
            longArrayOf(1, D1NaFlex.MAX_PATCHES.toLong(), D1NaFlex.PATCH_DIM.toLong())).use { pixels ->
            OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(D1NaFlex.MAX_PATCHES) { 1L }),
                longArrayOf(1, D1NaFlex.MAX_PATCHES.toLong())).use { mask ->
                OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(D1NaFlex.GRID.toLong(), D1NaFlex.GRID.toLong())),
                    longArrayOf(1, 2)).use { shapes ->
                        vision.run(mapOf("pixel_values" to pixels, "pixel_attention_mask" to mask,
                            "spatial_shapes" to shapes)).use { out ->
                            val tensor = out.get("image_features").get() as OnnxTensor
                            return FloatArray(tensor.floatBuffer.remaining()).also {
                                tensor.floatBuffer.get(it)
                            }
                        }
                    }
            }
        }
    }

    fun decide(features: FloatArray): JSONObject {
        val encoded = D1Prompt.encode(
            encode = { text -> tokenizer.encode(text) },
            idOf = { token -> if (token == "<bos>") bosTokenId else tokenizer.specialId(token) },
            state = V0_STATE,
            instructions = V0_QUESTION,
            options = V0_OPTIONS,
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
            .put("text_tokens", encoded.ids.size)
    }

    fun close() = listOf(vision, embeddings, decision).forEach { runCatching { it.close() } }
}

fun main(args: Array<String>) {
    val probes = StyleSheets().buildProbes()
    if (args.firstOrNull() == "dump") {
        val dir = File(args.getOrNull(1) ?: "style-seed-probes").apply { mkdirs() }
        probes.forEach { javax.imageio.ImageIO.write(it.image, "png", File(dir, "${it.id}.png")) }
        println("dumped ${probes.size} sheets to ${dir.absolutePath}")
        return
    }
    val modelDir = File(args.firstOrNull() ?: error("usage: Main <d1-model-dir> [out.json]"))
    val outFile = File(args.getOrNull(1) ?: "style-seed-eval.json")
    println("model dir: $modelDir")
    println("probes: ${probes.size} (bias=${probes.count { it.kind == "bias" }}, " +
        "discrimination=${probes.count { it.kind == "discrimination" }})")

    val engine = Engine(modelDir)
    val sheetsJson = JSONArray()
    val timings = ArrayList<Long>()
    try {
        probes.forEachIndexed { i, probe ->
            val argb = probe.image.getRGB(0, 0, 512, 512, null, 0, 512)
            val t0 = System.currentTimeMillis()
            val features = engine.visionFeatures(argb)
            val decision = engine.decide(features)
            timings += System.currentTimeMillis() - t0
            val correct = probe.goodPanel != null &&
                decision.optInt("selected", -1) == probe.goodPanel
            sheetsJson.put(JSONObject()
                .put("id", probe.id)
                .put("style", probe.style?.name ?: JSONObject.NULL)
                .put("kind", probe.kind)
                .put("good_panel", probe.goodPanel?.let { "${'A' + it}" } ?: JSONObject.NULL)
                .put("decision", decision.put("correct", correct)))
            println("[${i + 1}/${probes.size}] ${probe.id} " +
                "probs=${decision.getJSONArray("probs")} sel=${decision.opt("selected")} correct=$correct")
        }
    } finally {
        engine.close()
    }

    val rows = (0 until sheetsJson.length()).map { sheetsJson.getJSONObject(it) }
    val discRows = rows.filter { it.getString("kind") == "discrimination" }
    val biasRows = rows.filter { it.getString("kind") == "bias" }

    val correct = discRows.count { it.getJSONObject("decision").getBoolean("correct") }
    val gateCorrect = discRows.count {
        it.getJSONObject("decision").getBoolean("gate_pass") &&
            it.getJSONObject("decision").getBoolean("correct")
    }
    val byPosition = JSONObject()
    for (p in 0..2) {
        val subset = discRows.filter { it.getString("good_panel") == "${'A' + p}" }
        byPosition.put("${'A' + p}", subset.count { it.getJSONObject("decision").getBoolean("correct") })
    }
    val byStyle = JSONObject()
    for (style in Style.entries) {
        val subset = discRows.filter { it.optString("style") == style.name }
        byStyle.put(style.name, subset.count { it.getJSONObject("decision").getBoolean("correct") })
    }
    val argmaxDistribution = JSONObject()
    for (p in 0..2) {
        argmaxDistribution.put("${'A' + p}", discRows.count {
            it.getJSONObject("decision").optInt("selected", -1) == p
        })
    }
    val biasMax = biasRows.map {
        it.getJSONObject("decision").getJSONArray("probs").toList()
            .filterIsInstance<Double>().max()
    }

    val report = JSONObject()
        .put("purpose", "T10 candidate-diversity probe: does d1 follow composition grammar " +
            "when the three crops of one frame carry different styles (style-seed injection)")
        .put("model", MODEL_PIN)
        .put("prompt", JSONObject().put("version", "comp-choice-v0")
            .put("state", V0_STATE).put("question", V0_QUESTION)
            .put("options", JSONArray(V0_OPTIONS)))
        .put("runtime", "onnxruntime 1.22.0, desktop JVM, 4 threads")
        .put("date", "2026-10-18")
        .put("decision_rule", "argmax correct >=12/18 = content following; ~9/18 with bias " +
            "max-prob >=0.40 = positional responder, style seeding is diversity-only and the " +
            "0.10 gate stays; <4/18 between style groups = no observable difference")
        .put("sheets", sheetsJson)
        .put("summary", JSONObject()
            .put("argmax_correct", "$correct/${discRows.size}")
            .put("gate_selected_correct", "$gateCorrect/${discRows.size}")
            .put("correct_by_good_panel_position", byPosition)
            .put("correct_by_style", byStyle)
            .put("argmax_position_distribution", argmaxDistribution)
            .put("bias_mean_max_prob", Math.round(biasMax.average() * 10000.0) / 10000.0))
        .put("mean_ms_per_sheet", timings.average().toLong())
    outFile.writeText(report.toString(2))
    println("\n== summary: ${report.getJSONObject("summary")}")
    println("wrote $outFile")
}

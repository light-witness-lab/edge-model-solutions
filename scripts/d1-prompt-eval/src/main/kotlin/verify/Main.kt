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
import java.awt.geom.Ellipse2D
import java.awt.geom.Rectangle2D
import java.awt.image.BufferedImage
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import org.json.JSONArray
import org.json.JSONObject

/**
 * Isolated prompt-revision evaluation for the composition choice task (T07).
 *
 * The probe set is programmatic (fixed, no RNG, never used to tune the rule layer):
 *  - bias probes P1/P2: identical panels -> a position-neutral model stays near uniform;
 *  - discrimination probes: one GOOD panel (whole subject on a thirds point, breathing
 *    room) vs CUT (subject sliced by the photo edge) vs CORNER (only a sliver in a
 *    corner); the good panel rotates through A/B/C, so accuracy that follows content
 *    across physical positions is the permutation check.
 *
 * v0 is the shipped wording ("Which crop gives the best composition?" / "crop option A"
 * / empty state); v1 is D1Prompt's current candidate revision. Vision features are
 * computed once per sheet and shared by both prompt decisions.
 *
 * Pre-registered decision rule (set before seeing any number):
 *  1. adopt v1 only if v1 accuracy >= v0 accuracy on the 18 discrimination sheets;
 *  2. and v1 bias (mean max-prob on P1/P2) is no more than 0.05 above v0;
 *  3. a difference of fewer than 4/18 is reported as "no observable difference".
 * The probes test sensitivity to framing signals, not aesthetic truth; loaded-only
 * evidence stays loaded-only regardless of the outcome.
 */

private const val MODEL_PIN = "onnx-community/d1-omni-600M-ONNX@4ebc1b9 (decision_q4)"
private const val MIN_MARGIN = 0.10

// Both prompt revisions are inlined on purpose: this is a frozen experiment snapshot,
// it must keep producing v0/v1 numbers regardless of the app's current D1Prompt.
// v0: exact wording the app shipped before T07 (and after it — v1 was rejected).
private const val V0_STATE = ""
private const val V0_QUESTION = "Which crop gives the best composition?"
private val V0_OPTIONS = listOf("a: crop option A", "b: crop option B", "c: crop option C")

// v1: the criteria-rich candidate tested on 2026-10-18 (comp-choice-v1), rejected.
private const val V1_STATE = "Three crop options of one photograph, shown as panels labelled A, B and C."
private const val V1_QUESTION =
    "Which panel has the best composition? Prefer a clear main subject, balanced " +
        "placement on the thirds or a centered symmetric subject, breathing room " +
        "around the subject, and no important object cut by the frame edge."
private val V1_OPTIONS = listOf(
    "A: crop shown in panel A",
    "B: crop shown in panel B",
    "C: crop shown in panel C",
)

private data class Probe(val id: String, val kind: String, val goodPanel: Int?, val image: BufferedImage)

private class SheetBuilder {
    // Mirrors CandidateContactSheet geometry: 512 canvas, rgb(244) ground, 256 quadrants,
    // label band 0..40, photo rect (x+8, y+40)-(x+248, y+248), 1px rgb(210) border.
    private val ground = Color(244, 244, 244)
    private val border = Color(210, 210, 210)
    private val labelColor = Color.BLACK
    private val photoBg = Color(176, 176, 176)
    private val photoBgLow = Color(150, 150, 150)

    private val photoRects = listOf(
        Rectangle2D.Float(8f, 40f, 240f, 208f),
        Rectangle2D.Float(264f, 40f, 240f, 208f),
        Rectangle2D.Float(8f, 296f, 240f, 208f),
    )

    private fun newSheet(): BufferedImage {
        val img = BufferedImage(512, 512, BufferedImage.TYPE_INT_RGB)
        val g = img.createGraphics()
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
        g.color = ground
        g.fillRect(0, 0, 512, 512)
        g.font = Font(Font.SANS_SERIF, Font.BOLD, 30)
        g.color = labelColor
        listOf("A" to 128f, "B" to 384f, "C" to 128f).forEachIndexed { i, (letter, cx) ->
            val y = if (i < 2) 0f else 256f
            g.drawCentered(letter, cx, y + 31f)
        }
        return img.also { g.dispose() }
    }

    private fun Graphics2D.drawPhoto(panel: Int, content: (Graphics2D, Rectangle2D.Float) -> Unit) {
        val r = photoRects[panel]
        color = photoBg
        fill(r)
        content(this, r)
        color = border
        stroke = BasicStroke(1f)
        draw(r)
    }

    /** Simple "photograph": mid-gray scene, optional horizon, one high-contrast subject. */
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
        val shape = if (round) {
            Ellipse2D.Float(cx - radius, cy - radius, radius * 2, radius * 2)
        } else {
            Rectangle2D.Float(cx - radius, cy - radius, radius * 2, radius * 2)
        }
        g.fill(shape)
    }

    private fun goodContent(scene: Int): (Graphics2D, Rectangle2D.Float) -> Unit {
        // Whole subject on a thirds point, 34px radius, never touching the edge.
        val thirdsX = floatArrayOf(1f / 3f, 2f / 3f, 1f / 3f, 2f / 3f)
        val thirdsY = floatArrayOf(1f / 3f, 1f / 3f, 2f / 3f, 2f / 3f)
        val tx = thirdsX[scene % 4]; val ty = thirdsY[scene % 4]
        val light = scene % 2 == 0
        val subject = if (light) Color(232, 214, 140) else Color(58, 60, 74)
        val round = scene % 2 == 0
        val horizon = scene % 3 != 0
        return { g, r ->
            paintScene(g, r, subject, round, horizon,
                r.x + r.width * tx, r.y + r.height * ty, 34f)
        }
    }

    /** Subject centered on the photo's vertical edge: half of it is cut off. */
    private fun cutContent(scene: Int): (Graphics2D, Rectangle2D.Float) -> Unit {
        val subject = if (scene % 2 == 0) Color(232, 214, 140) else Color(58, 60, 74)
        val round = scene % 2 == 1
        val cutRight = scene % 2 == 0
        return { g, r ->
            paintScene(g, r, subject, round, horizon = true,
                if (cutRight) r.x + r.width else r.x, r.y + r.height * 0.5f, 40f)
        }
    }

    /** Subject centered on a photo corner: only a quarter sliver remains. */
    private fun cornerContent(scene: Int): (Graphics2D, Rectangle2D.Float) -> Unit {
        val subject = if (scene % 3 == 0) Color(232, 214, 140) else Color(58, 60, 74)
        val corners = arrayOf(
            floatArrayOf(0f, 0f), floatArrayOf(1f, 0f), floatArrayOf(0f, 1f), floatArrayOf(1f, 1f),
        )
        val c = corners[scene % 4]
        return { g, r ->
            paintScene(g, r, subject, round = true, horizon = false,
                r.x + r.width * c[0], r.y + r.height * c[1], 52f)
        }
    }

    private fun emptyContent(): (Graphics2D, Rectangle2D.Float) -> Unit = { _, _ -> }

    fun buildProbes(): List<Probe> {
        val probes = ArrayList<Probe>()

        // Bias: identical panels, no information to distinguish them.
        probes += Probe("P1-uniform", "bias", null, newSheet().also { img ->
            val g = img.createGraphics()
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
            for (p in 0..2) g.drawPhoto(p, emptyContent())
            g.dispose()
        })
        probes += Probe("P2-same-scene", "bias", null, newSheet().also { img ->
            val g = img.createGraphics()
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
            for (p in 0..2) g.drawPhoto(p, goodContent(scene = 0))
            g.dispose()
        })

        // Discrimination: good panel rotates A/B/C across six scene variants.
        for (scene in 0..5) {
            for (goodPanel in 0..2) {
                val sheet = newSheet()
                val g = sheet.createGraphics()
                g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
                var badKinds = 0
                for (panel in 0..2) {
                    val content = if (panel == goodPanel) goodContent(scene)
                    else if (badKinds++ % 2 == 0) cutContent(scene) else cornerContent(scene)
                    g.drawPhoto(panel, content)
                }
                g.dispose()
                probes += Probe("D-s${scene}-good${'A' + goodPanel}", "discrimination", goodPanel, sheet)
            }
        }
        return probes
    }

    // Graphics2D has no built-in centered text; tiny helper matching Paint.Align.CENTER.
    private fun Graphics2D.drawCentered(text: String, cx: Float, baselineY: Float) {
        val w = fontMetrics.stringWidth(text)
        drawString(text, cx - w / 2f, baselineY)
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

    fun decide(features: FloatArray, state: String, question: String, options: List<String>): JSONObject {
        val encoded = D1Prompt.encode(
            encode = { text -> tokenizer.encode(text) },
            idOf = { token -> if (token == "<bos>") bosTokenId else tokenizer.specialId(token) },
            state = state,
            instructions = question,
            options = options,
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
    val modelDir = File(args.firstOrNull() ?: error("usage: Main <d1-model-dir> [out.json]"))
    val outFile = File(args.getOrNull(1) ?: "prompt-eval.json")
    println("model dir: $modelDir")

    val probes = SheetBuilder().buildProbes()
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
            val v0 = engine.decide(features, V0_STATE, V0_QUESTION, V0_OPTIONS)
            val v1 = engine.decide(features, V1_STATE, V1_QUESTION, V1_OPTIONS)
            timings += System.currentTimeMillis() - t0
            val correct = fun(decision: JSONObject): Boolean =
                probe.goodPanel != null && decision.optInt("selected", -1) == probe.goodPanel
            sheetsJson.put(JSONObject()
                .put("id", probe.id).put("kind", probe.kind)
                .put("good_panel", probe.goodPanel?.let { "${'A' + it}" } ?: JSONObject.NULL)
                .put("v0", v0.put("correct", correct(v0)))
                .put("v1", v1.put("correct", correct(v1))))
            println("[${i + 1}/${probes.size}] ${probe.id} " +
                "v0=${v0.getJSONArray("probs")} sel=${v0.opt("selected")} " +
                "v1=${v1.getJSONArray("probs")} sel=${v1.opt("selected")}")
        }
    } finally {
        engine.close()
    }

    val sheetList = (0 until sheetsJson.length()).map { sheetsJson.getJSONObject(it) }

    fun summarize(version: String): JSONObject {
        val rows = sheetList.map { it.getJSONObject(version) }
        val biasRows = sheetList.filter { it.getString("kind") == "bias" }
            .map { it.getJSONObject(version) }
        val discRows = sheetList.filter { it.getString("kind") == "discrimination" }
        val biasMax = biasRows.map { it.getJSONArray("probs").toList().filterIsInstance<Double>().max() }
        val acc = discRows.count { it.getJSONObject(version).getBoolean("correct") }
        val byPosition = JSONObject()
        for (p in 0..2) {
            val subset = discRows.filter { it.getString("good_panel") == "${'A' + p}" }
            byPosition.put("${'A' + p}", subset.count { it.getJSONObject(version).getBoolean("correct") })
        }
        val gatePass = rows.count { it.getBoolean("gate_pass") }
        return JSONObject()
            .put("accuracy", "$acc/${discRows.size}")
            .put("accuracy_rate", Math.round(acc * 10000.0 / discRows.size) / 10000.0)
            .put("correct_by_good_panel_position", byPosition)
            .put("bias_mean_max_prob", Math.round(biasMax.average() * 10000.0) / 10000.0)
            .put("gate_pass_rate", "$gatePass/${rows.size}")
    }

    val report = JSONObject()
        .put("purpose", "T07 isolated prompt-revision probe; framing-signal sensitivity, not aesthetic truth")
        .put("model", MODEL_PIN)
        .put("runtime", "onnxruntime 1.22.0, desktop JVM, 4 threads")
        .put("date", "2026-10-18")
        .put("prompts", JSONObject()
            .put("v0", JSONObject().put("state", V0_STATE).put("question", V0_QUESTION)
                .put("options", JSONArray(V0_OPTIONS)))
            .put("v1", JSONObject()
                .put("version", "comp-choice-v1")
                .put("state", V1_STATE)
                .put("question", V1_QUESTION)
                .put("options", JSONArray(V1_OPTIONS))))
        .put("decision_rule", "adopt v1 iff v1.accuracy>=v0.accuracy over 18 sheets and " +
            "v1.bias_mean_max_prob<=v0+0.05; <4/18 difference = no observable difference")
        .put("sheets", sheetsJson)
        .put("summary_v0", summarize("v0"))
        .put("summary_v1", summarize("v1"))
        .put("mean_ms_per_sheet_both_decisions", timings.average().toLong())
    outFile.writeText(report.toString(2))
    println("\n== summary v0: ${report.getJSONObject("summary_v0")}")
    println("== summary v1: ${report.getJSONObject("summary_v1")}")
    println("wrote $outFile")
}

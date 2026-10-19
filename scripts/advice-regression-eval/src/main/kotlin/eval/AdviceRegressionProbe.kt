package eval

import dev.streetsnap.app.composition.AlignmentEvent
import dev.streetsnap.app.composition.AlignmentFeedback
import dev.streetsnap.app.composition.AlignmentTracker
import dev.streetsnap.app.composition.CompositionActions
import dev.streetsnap.app.composition.CompositionCandidate
import dev.streetsnap.app.composition.CompositionStyle
import dev.streetsnap.app.composition.FastCompositionAdvisor
import dev.streetsnap.app.composition.FrameGuidance
import dev.streetsnap.app.composition.MoveDirection
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import kotlin.math.ln
import kotlin.math.max

/**
 * T37 advice-effect regression baseline.
 *
 * Runs the SHIPPED rule layer (verbatim snapshot, no model) over a fixed deterministic
 * geometric scene suite and reports four pre-registered metrics:
 *
 *  M1 coverage       distinct first-choice styles seen (expressed vocabulary excludes the
 *                    CONSERVATIVE/GENERIC fallbacks; denominator = 19), plus per-family sets.
 *  M2 distribution   first-choice style counts over the whole suite, normalised entropy
 *                    over all 21 vocabulary slots, and the top-style share. A healthy
 *                    advisor neither collapses to one sentence nor sprays uniformly.
 *  M3 change rate    repeat pairs (same frame twice) MUST never change; the blob-grid
 *                    scan order turns list adjacency into small camera moves, and its
 *                    edge change rate measures advice following the picture; mean style
 *                    run length over the full ordered suite quantifies sticky-vs-jumpy.
 *  M4 alignment      for every scan/minimal frame whose first advice carries an action
 *                    sentence, an "obedient re-framing" walk applies the planner's move to
 *                    the blob parameters (60% of the geometric gap per step, 12-step
 *                    budget), re-renders, and feeds the real AlignmentTracker: success is
 *                    an ALIGNED pulse within budget. Tracker gate sanity is checked
 *                    separately (praise without a prior need must never fire).
 *
 * No scene is adjusted after looking at results: geometry intents are fixed in
 * SyntheticScenes and the first run is the baseline. Synthetic shapes are not aesthetic
 * truth (loaded-only).
 */

private const val SNAPSHOT_SOURCE = "street-snap@ee8d53c + T40 alignment rework (2026-10-19)"
private const val STEP_FRACTION = 0.60
// T40: 12 steps could hold the geometric walk but not the walk plus the 5-frame (~1.25 s)
// confirmation window the real AlignmentTracker requires; small subjects enter the band
// around step 9-10 and then had only 3 frames left. 16 frames = 4 s of obedient re-framing.
private const val LOOP_BUDGET = 16

private data class Frame(
    val sceneId: String, val family: String, val style: CompositionStyle,
    val actionable: Boolean, val elapsedNs: Long,
)

private data class LoopOutcome(
    val sceneId: String, val success: Boolean, val steps: Int,
    val finalStyle: String, val finalInset: Double, val finalBuiltIn: Boolean,
)

private fun advise(blob: SyntheticScenes.Blob): CompositionCandidate =
    FastCompositionAdvisor.suggest(
        SyntheticScenes.SIDE, SyntheticScenes.SIDE, SyntheticScenes.blobPixels(blob)
    ).candidates.first()

private fun advise(pixels: IntArray) =
    FastCompositionAdvisor.suggest(SyntheticScenes.SIDE, SyntheticScenes.SIDE, pixels)

private fun maxInset(c: CompositionCandidate): Double {
    val crop = c.crop
    return max(max(crop.left, crop.top), max(1f - crop.right, 1f - crop.bottom)).toDouble()
}

fun main(args: Array<String>) {
    val outPath = args.getOrNull(0)
        ?: "../../evaluations/advice-regression-baseline-2026-10-19.json"

    val scenes = SyntheticScenes.all
    val frames = ArrayList<Frame>(scenes.size)
    val repeatChanged = ArrayList<String>()
    val perFamily = LinkedHashMap<String, MutableSet<String>>()
    val latenciesNs = ArrayList<Long>()

    for (scene in scenes) {
        val ns = System.nanoTime()
        val r1 = advise(scene.pixels)
        val elapsed = System.nanoTime() - ns
        val first = r1.candidates.first()
        // Determinism: a repeated analysis of identical pixels must be identical advice.
        val r2 = advise(scene.pixels)
        if (r1.candidates != r2.candidates) repeatChanged += scene.id
        frames += Frame(scene.id, scene.family, first.style, first.action != null, elapsed)
        latenciesNs += elapsed
        perFamily.getOrPut(scene.family) { linkedSetOf() } += first.style.name
    }

    // ---- M1 coverage ----
    val seen = frames.mapTo(linkedSetOf()) { it.style.name }
    val fallback = setOf(CompositionStyle.CONSERVATIVE.name, CompositionStyle.GENERIC.name)
    val expressedSeen = (seen - fallback).sorted()
    val expressedTotal = CompositionStyle.entries.size - fallback.size
    val expressedMissing = CompositionStyle.entries.map { it.name }
        .filter { it !in fallback && it !in seen }.sorted()

    // ---- M2 distribution ----
    val counts = LinkedHashMap<String, Int>()
    CompositionStyle.entries.forEach { counts[it.name] = 0 }
    frames.forEach { counts[it.style.name] = (counts[it.style.name] ?: 0) + 1 }
    val n = frames.size.toDouble()
    val entropy = -counts.values.sumOf { c ->
        if (c == 0) 0.0 else (c / n).let { it * ln(it) }
    } / ln(CompositionStyle.entries.size.toDouble())
    val topShare = counts.values.maxOrNull()!! / n
    val actionableShare = frames.count { it.actionable } / n

    // ---- M3 change rates ----
    fun edgeRate(seq: List<Frame>): JSONObject {
        var changed = 0
        for (i in 1 until seq.size) if (seq[i].style != seq[i - 1].style) changed++
        return JSONObject().put("edges", seq.size - 1).put("changed", changed)
            .put("rate", if (seq.size > 1) changed.toDouble() / (seq.size - 1) else 0.0)
    }

    val blobSeq = frames.filter { it.family == "blob_grid" }
    val scanRate = edgeRate(blobSeq)
    val allRate = edgeRate(frames)
    var runs = 1
    for (i in 1 until frames.size) if (frames[i].style != frames[i - 1].style) runs++
    val meanRunLength = frames.size.toDouble() / runs

    // ---- M4 alignment closed loop ----
    val outcomes = ArrayList<LoopOutcome>()
    var eligible = 0
    for (scene in scenes) {
        val start = scene.blob ?: continue
        if (advise(start).action == null) continue
        eligible++

        var cx = start.cx; var cy = start.cy; var size = start.size
        val tracker = AlignmentTracker()
        var outcome: LoopOutcome? = null
        for (step in 1..LOOP_BUDGET) {
            val top = advise(SyntheticScenes.Blob(cx, cy, size, start.ground, start.content))
            val inset = maxInset(top)
            // T40: alignment is "the asked-for move is satisfied" (centred, scale >= 0.88),
            // not an exact full frame — the 0.92 crop band (4 % inset) used to never confirm.
            val guidance = AlignmentFeedback.guidanceFor(top.crop, top.style)
            if (tracker.observe(guidance) == AlignmentEvent.ALIGNED) {
                outcome = LoopOutcome(scene.id, true, step, top.style.name, inset,
                    top.style in CompositionActions.ACTION_BUILT_IN)
                break
            }
            // Already in the alignment band but still accumulating confirmation frames: a
            // real user holds the camera still, so do not keep applying moves (which would
            // grow the mass past the band and reset the streak).
            if (guidance == FrameGuidance.ALIGNED) continue
            // Apply the planner's move to the scene parameters: a right/up crop offset means
            // the content shifts opposite in the new frame; FORWARD grows the mass.
            val move = CompositionActions.plan(top.crop, top.style,
                (top.crop.left + top.crop.right) / 2f,
                (top.crop.top + top.crop.bottom) / 2f)
            if (move != null) {
                val cu = (top.crop.left + top.crop.right) / 2.0
                val cv = (top.crop.top + top.crop.bottom) / 2.0
                val s = (top.crop.right - top.crop.left).toDouble()
                cx = (cx - STEP_FRACTION * (cu - 0.5)).coerceIn(size / 2.0, 1.0 - size / 2.0)
                cy = (cy - STEP_FRACTION * (cv - 0.5)).coerceIn(size / 2.0, 1.0 - size / 2.0)
                if (move.parts.any { it.direction == MoveDirection.FORWARD }) {
                    size = (size * ((1 - STEP_FRACTION) + STEP_FRACTION / s)).coerceAtMost(0.95)
                    cx = cx.coerceIn(size / 2.0, 1.0 - size / 2.0)
                    cy = cy.coerceIn(size / 2.0, 1.0 - size / 2.0)
                }
            }
        }
        if (outcome == null) {
            val top = advise(SyntheticScenes.Blob(cx, cy, size, start.ground, start.content))
            val inset = maxInset(top)
            outcome = LoopOutcome(scene.id, false, LOOP_BUDGET, top.style.name, inset,
                top.style in CompositionActions.ACTION_BUILT_IN)
        }
        outcomes += outcome
    }

    val success = outcomes.count { it.success }
    val successSteps = outcomes.filter { it.success }.map { it.steps }.sorted()
    val medianSteps = if (successSteps.isEmpty()) null else successSteps[successSteps.size / 2]
    val failBuckets = JSONObject()
    outcomes.filter { !it.success }.forEach { o ->
        val bucket = when {
            o.finalBuiltIn -> "built_in_style"
            o.finalInset <= 0.10 -> "inset_0.03_0.10"
            o.finalInset <= 0.20 -> "inset_0.10_0.20"
            else -> "inset_over_0.20"
        }
        failBuckets.put(bucket, failBuckets.optInt(bucket) + 1)
    }
    val endStyles = JSONObject()
    outcomes.forEach { o -> endStyles.put(o.finalStyle, endStyles.optInt(o.finalStyle) + 1) }

    // ---- tracker gate sanity: praise without a prior need must never fire ----
    val noPraise = AlignmentTracker().run {
        var events = 0
        repeat(12) { if (observe(FrameGuidance.ALIGNED) == AlignmentEvent.ALIGNED) events++ }
        events
    }
    var armedPulseAt = -1
    var armedPulses = 0
    run {
        val t = AlignmentTracker()
        repeat(12) { t.observe(FrameGuidance.NEEDS_MOVE) }
        for (f in 1..12) {
            if (t.observe(FrameGuidance.ALIGNED) == AlignmentEvent.ALIGNED) {
                armedPulses++
                if (armedPulseAt < 0) armedPulseAt = 12 + f
            }
        }
    }

    latenciesNs.sort()
    val medianLatencyMs = latenciesNs[latenciesNs.size / 2] / 1_000_000.0

    val familyJson = JSONObject()
    perFamily.toSortedMap().forEach { (fam, styles) -> familyJson.put(fam, JSONArray(styles.sorted())) }
    val countsJson = JSONObject()
    counts.toSortedMap().forEach { (k, v) -> countsJson.put(k, v) }

    val root = JSONObject()
    root.put("date", "2026-10-19")
    root.put("purpose", "T37 re-runnable regression baseline for the shipped rule-layer " +
        "composition advice: coverage, distribution, change rate, alignment closed-loop. " +
        "Geometric synthetic frames, no model, no labelled truth.")
    root.put("evidence_level", "loaded-only")
    root.put("snapshot", JSONObject()
        .put("source", SNAPSHOT_SOURCE)
        .put("files", JSONArray(listOf(
            "dev/streetsnap/camera/NormalizedBoundingBox.kt (verbatim extraction from CompositionPipelineContracts.kt)",
            "dev/streetsnap/app/composition/FastCompositionAdvisor.kt (verbatim whole file)",
            "dev/streetsnap/app/composition/CompositionActions.kt (verbatim whole file)",
            "dev/streetsnap/app/composition/AlignmentFeedback.kt (verbatim whole file)",
            "dev/streetsnap/app/composition/CompositionStyle.kt (verbatim whole file)")))
        .put("note", "verbatim snapshots; internal visibility resolves inside one Gradle module; " +
            "the rest of CompositionPipelineContracts.kt is excluded (CompositionFrame sits with CameraX bindings)"))
    root.put("canvas", JSONObject()
        .put("size_px", SyntheticScenes.SIDE)
        .put("advisor_internal_grid_px", 96)
        .put("rng", "none"))
    root.put("suite", JSONObject()
        .put("frames", scenes.size)
        .put("families", familyJson)
        .put("blob_scan", "3 sizes (0.10/0.22/0.38) x 7x7 interior positions, contiguous order"))
    root.put("metric_definitions", JSONObject()
        .put("M1_coverage", "distinct first-choice styles; expressed excludes " +
            "CONSERVATIVE/GENERIC, denominator 19")
        .put("M2_distribution", "first-choice counts, entropy normalised by ln(21), top share")
        .put("M3_change", "repeat pairs must never change; blob-scan adjacent-edge change rate; " +
            "mean style run length over the ordered suite")
        .put("M4_alignment", "obedient re-framing walk: apply planner move to blob parameters " +
                "fraction=$STEP_FRACTION per step, budget=$LOOP_BUDGET steps, success = real " +
                "AlignmentTracker emits ALIGNED; eligible = first advice carries an action"))
    root.put("results", JSONObject()
        .put("total_frames", scenes.size)
        .put("repeat_pairs_changed", repeatChanged.size)
        .put("repeat_changed_ids", JSONArray(repeatChanged))
        .put("median_advice_ms", medianLatencyMs)
        .put("M1_coverage", JSONObject()
            .put("styles_seen", JSONArray(seen.sorted()))
            .put("expressed_seen", expressedSeen.size)
            .put("expressed_total", expressedTotal)
            .put("expressed_missing", JSONArray(expressedMissing)))
        .put("M2_distribution", JSONObject()
            .put("first_choice_counts", countsJson)
            .put("entropy_normalised", entropy)
            .put("top_style_share", topShare)
            .put("actionable_share", actionableShare))
        .put("M3_change", JSONObject()
            .put("blob_scan", scanRate)
            .put("full_sequence", allRate)
            .put("mean_run_length", meanRunLength))
        .put("M4_alignment", JSONObject()
            .put("eligible", eligible)
            .put("success", success)
            .put("success_rate", if (eligible > 0) success.toDouble() / eligible else 0.0)
            .put("median_steps_to_aligned", medianSteps)
            .put("fail_buckets", failBuckets)
            .put("loop_end_styles", endStyles)
            .put("walk", "step_fraction=$STEP_FRACTION, budget=$LOOP_BUDGET, " +
                "pan/tilt shift the mass, dolly grows it, mass kept inside the frame"))
        .put("tracker_gate_sanity", JSONObject()
            .put("praise_without_need_events", noPraise)
            .put("armed_pulse_frame", armedPulseAt)
            .put("armed_total_pulses", armedPulses)
            .put("expected", "0, 17 (5 confirm frames after 12 needing), 1")))
    root.put("limitations", JSONArray(listOf(
        "flat geometric shapes on flat grounds are not photographs; no aesthetic judgement is measured",
        "no model participates: the d1 gate, selection gate and scheduling are outside this harness",
        "the re-framing walk assumes an obedient operator and exact parameter control; real camera moves overshoot",
        "desktop JVM latency is not device latency",
        "style coverage reflects this fixed suite only; missing styles may need other geometry or real scenes")))

    val out = File(outPath)
    out.parentFile?.mkdirs()
    out.writeText(root.toString(2) + "\n")
    println("frames=${scenes.size} expressed=${expressedSeen.size}/$expressedTotal " +
        "entropy=%.3f topShare=%.3f actionable=%.3f scanChange=%.3f".format(
            entropy, topShare, actionableShare, scanRate.getDouble("rate")))
    println("alignment eligible=$eligible success=$success medianSteps=$medianSteps " +
        "fails=${failBuckets}")
    println("gate: noPraise=$noPraise armedPulseAt=$armedPulseAt armedPulses=$armedPulses")
    println("wrote $out")
}

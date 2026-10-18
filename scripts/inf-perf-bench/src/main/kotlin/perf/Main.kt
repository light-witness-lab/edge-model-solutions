package perf

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONObject
import java.io.File
import java.time.LocalDate
import kotlin.math.sin
import kotlin.system.exitProcess

/**
 * Inference performance harness skeleton for benchmarks/inference-perf.
 *
 * Two modes:
 *   --selfcheck OUT
 *       Exercises the warm-up / timing / percentile / JSON plumbing with a deterministic
 *       CPU workload. Output is marked workload=synthetic-selfcheck and is NEVER a model
 *       result; it only proves the measurement pipeline runs.
 *
 *   --model DIR --workload W1 [--threads N] --out OUT
 *       Pins ORT 1.22, enumerates *.onnx under DIR, and times one cold OrtSession creation
 *       per graph (real load latency, no input tensor needed). Forward segments stay null
 *       until the NaFlex vision layout, prompt encoder and decision softmax are bound by
 *       reusing the frozen snapshots in scripts/d1-prompt-eval. No forward number is
 *       fabricated.
 *
 * Evidence level is always not-verified here. Promotion follows benchmarks/inference-perf.
 */

private const val SCHEMA = "inference-perf/v1"

/**
 * Per-segment wall-clock samples. Warm-up iterations are dropped by the caller
 * (the recorder only stores iterations whose index is >= warmup).
 */
private class RunTimer(private val warmup: Int) {
    private val raw = LinkedHashMap<String, MutableList<Double>>()
    private var iteration = 0

    inline fun measure(name: String, block: () -> Unit) {
        val start = System.nanoTime()
        block()
        if (iteration >= warmup) {
            raw.getOrPut(name) { mutableListOf() }.add((System.nanoTime() - start) / 1_000_000.0)
        }
    }

    fun nextIteration() { iteration++ }

    fun segmentsJson(): JSONObject {
        val out = JSONObject()
        for ((name, values) in raw) {
            val s = values.sorted()
            fun pct(q: Double) = s[(Math.round(q * (s.size - 1)).toInt()).coerceIn(0, s.size - 1)]
            fun r(v: Double) = Math.round(v * 1000.0) / 1000.0
            out.put(name, JSONObject()
                .put("p50", r(pct(0.50))).put("p95", r(pct(0.95)))
                .put("min", r(s.first())).put("max", r(s.last()))
                .put("mean", r(s.average())).put("n", s.size))
        }
        return out
    }

    fun mean(name: String) = raw[name]?.average() ?: 0.0
}

/** Deterministic, non-trivial CPU work whose result is consumed so the JIT cannot elide it. */
private fun fillAndSort(n: Int, seed: Int): Double {
    val a = DoubleArray(n) { i -> sin((i + seed).toDouble() * 0.0009) * (1.0 + (i % 7)) }
    a.sort()
    var sum = 0.0
    var i = 0
    while (i < n) { sum += a[i]; i += 997 }
    return sum
}

private fun timedRun(warmup: Int, iters: Int): Pair<RunTimer, Double> {
    val rec = RunTimer(warmup)
    var checksum = 0.0
    for (k in 0 until warmup + iters) {
        var v = 0.0; var e = 0.0; var d = 0.0
        rec.measure("end_to_end_round") {
            rec.measure("vision_encode") { v = fillAndSort(300_000, k) }
            rec.measure("prompt_embed") { e = fillAndSort(20_000, k + 1) }
            rec.measure("decision_or_prefill") { d = fillAndSort(120_000, k + 2) }
        }
        checksum += v + e * 0.01 + d * 0.1
        rec.nextIteration()
    }
    return rec to checksum
}

private fun runSelfCheck(out: File, warmup: Int, iters: Int) {
    val (rec, checksum) = timedRun(warmup, iters)
    val e2eMean = rec.mean("end_to_end_round")
    val json = JSONObject()
        .put("schema", SCHEMA)
        .put("solution", "composition-assistant")
        .put("run_id", "selfcheck-${LocalDate.now()}")
        .put("date", LocalDate.now().toString())
        .put("evidence_level", "not-verified")
        .put("device", JSONObject().put("name", "desktop-jvm"))
        .put("workload", JSONObject()
            .put("id", "synthetic-selfcheck")
            .put("fixture", "deterministic in-memory sort, no model")
            .put("prefix_constant", JSONObject.NULL)
            .put("prefix_or_vision_shared", JSONObject.NULL)
            .put("generated_tokens", 0))
        .put("policy", JSONObject().put("warmup_iters", warmup).put("measured_iters", iters)
            .put("single_concurrency", true).put("deterministic_cpu", true))
        .put("segments_ms", rec.segmentsJson())
        .put("throughput", JSONObject()
            .put("rounds_per_sec", if (e2eMean > 0) Math.round(1000.0 / e2eMean * 1000.0) / 1000.0 else JSONObject.NULL))
        .put("output_validity", JSONObject.NULL)
        .put("resources", JSONObject.NULL)
        .put("notes", "Synthetic plumbing check (sort workload, checksum=${Math.round(checksum)}); " +
            "not a model measurement. Segment names mirror the real pipeline in benchmarks/inference-perf.")
    write(out, json)
}

private fun runModelProbe(modelDir: File, workload: String, threads: Int, out: File) {
    if (!modelDir.isDirectory) {
        System.err.println("model directory not found: $modelDir"); exitProcess(2)
    }
    val graphs = modelDir.walkTopDown().filter { it.isFile && it.name.endsWith(".onnx") }.sortedBy { it.name }.toList()
    if (graphs.isEmpty()) {
        System.err.println("no *.onnx under $modelDir"); exitProcess(2)
    }

    val env = OrtEnvironment.getEnvironment()
    val createMs = JSONObject()
    val errors = JSONObject()
    for (graph in graphs) {
        val opts = OrtSession.SessionOptions()
        opts.setIntraOpNumThreads(threads)
        try {
            val t0 = System.nanoTime()
            env.createSession(graph.toString(), opts).use { session ->
                val ms = (System.nanoTime() - t0) / 1_000_000.0
                createMs.put(graph.relativeTo(modelDir).path, Math.round(ms * 1000.0) / 1000.0)
            }
        } catch (t: Throwable) {
            errors.put(graph.relativeTo(modelDir).path, t.javaClass.simpleName + ": " + (t.message ?: ""))
        } finally {
            opts.close()
        }
    }

    fun nullSeg() = JSONObject().put("p50", JSONObject.NULL).put("p95", JSONObject.NULL)
        .put("min", JSONObject.NULL).put("max", JSONObject.NULL).put("mean", JSONObject.NULL)
    val segments = JSONObject()
        .put("vision_encode", nullSeg()).put("prompt_embed", nullSeg())
        .put("decision_or_prefill", nullSeg()).put("decode_per_token", nullSeg())
        .put("end_to_end_round", nullSeg())

    val json = JSONObject()
        .put("schema", SCHEMA)
        .put("solution", "composition-assistant")
        .put("run_id", "probe-${LocalDate.now()}-$workload-t$threads")
        .put("date", LocalDate.now().toString())
        .put("evidence_level", "not-verified")
        .put("model", JSONObject().put("id", modelDir.name).put("path", modelDir.absolutePath))
        .put("runtime", JSONObject().put("name", "onnxruntime").put("version", "1.22.0")
            .put("execution_provider", "cpu").put("intra_op_threads", threads))
        .put("workload", JSONObject().put("id", workload))
        .put("segments_ms", segments)
        .put("session_create_ms", createMs)
        .put("notes", "Cold OrtSession creation per graph is measured; forward latencies are null " +
            "until the NaFlex/vision, prompt-encoder and decision softmax inputs are bound from the " +
            "scripts/d1-prompt-eval snapshots. No forward number is fabricated.")
    if (errors.length() > 0) json.put("session_create_errors", errors)
    write(out, json)
}

private fun write(out: File, json: JSONObject) {
    out.parentFile?.mkdirs()
    out.writeText(json.toString(2) + "\n")
    println("wrote $out")
    println(json.toString(2))
}

fun main(args: Array<String>) {
    val opts = parseArgs(args)
    val warmup = (opts["--warmup"] ?: "3").toInt()
    val iters = (opts["--iters"] ?: "20").toInt()
    when {
        opts.containsKey("--selfcheck") ->
            runSelfCheck(File(opts["--selfcheck"]!!), warmup, iters)
        opts.containsKey("--model") ->
            runModelProbe(File(opts["--model"]!!), opts["--workload"] ?: "W1",
                (opts["--threads"] ?: "4").toInt(), File(opts["--out"] ?: "/tmp/inf-perf.json"))
        else -> {
            System.err.println("usage: --selfcheck OUT | --model DIR --workload W1 [--threads N] --out OUT " +
                "[--warmup 3] [--iters 20]")
            exitProcess(2)
        }
    }
}

private fun parseArgs(args: Array<String>): Map<String, String> {
    val m = LinkedHashMap<String, String>()
    var i = 0
    while (i < args.size) {
        val a = args[i]
        if (a.startsWith("--")) {
            if (i + 1 < args.size && !args[i + 1].startsWith("--")) { m[a] = args[i + 1]; i += 2 }
            else { m[a] = "true"; i += 1 }
        } else i += 1
    }
    return m
}

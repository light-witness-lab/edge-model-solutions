package eval

import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Deterministic 96×96 ARGB scenes for the advice regression baseline.
 *
 * There is no RNG and no photographic content: every scene is a geometric arrangement of
 * flat rectangles/bands on a flat ground. The advisor reduces its input to 96 px anyway,
 * so rendering at exactly 96×96 makes the statistics inspected identical to what the
 * shipped rule layer sees. The intents below describe what each arrangement is *designed*
 * to probe; the styles the advisor actually emits are recorded as the baseline, never
 * tuned to match the intent.
 */
object SyntheticScenes {
    const val SIDE = 96

    // High-contrast palette borrowed from the app's own advisor unit tests.
    const val DARK = 0xFF262626.toInt()
    const val LIGHT = 0xFFE6E6E6.toInt()
    const val MID = 0xFF707070.toInt()
    const val UPPER = 0xFF4A4A4A.toInt()
    const val LOWER = 0xFFA6A6A6.toInt()
    const val PALE_GROUND = 0xFFB8B8B8.toInt()
    const val HALF_DARK = 0xFF2E2E2E.toInt()
    const val HALF_LIGHT = 0xFFD2D2D2.toInt()

    /** Re-renderable single-blob state, used by the alignment closed-loop simulation. */
    data class Blob(val cx: Double, val cy: Double, var size: Double,
                    val ground: Int = DARK, val content: Int = LIGHT)

    data class Scene(
        val id: String,
        val family: String,
        val intent: String,
        val pixels: IntArray,
        val blob: Blob? = null,
    )

    private fun canvas(ground: Int) = IntArray(SIDE * SIDE) { ground }

    private fun fillRect(
        p: IntArray, cx: Double, cy: Double, sx: Double, sy: Double, color: Int,
    ) {
        val x0 = ((cx - sx / 2.0) * SIDE).toInt().coerceIn(0, SIDE)
        val x1 = ((cx + sx / 2.0) * SIDE).toInt().coerceIn(0, SIDE)
        val y0 = ((cy - sy / 2.0) * SIDE).toInt().coerceIn(0, SIDE)
        val y1 = ((cy + sy / 2.0) * SIDE).toInt().coerceIn(0, SIDE)
        for (y in y0 until y1) for (x in x0 until x1) p[y * SIDE + x] = color
    }

    private fun hSplit(p: IntArray, y: Double, upper: Int, lower: Int) {
        val cut = (y * SIDE).toInt().coerceIn(0, SIDE)
        for (i in p.indices) p[i] = if (i / SIDE < cut) upper else lower
    }

    private fun vSplit(p: IntArray, x: Double, left: Int, right: Int) {
        val cut = (x * SIDE).toInt().coerceIn(0, SIDE)
        for (i in p.indices) p[i] = if (i % SIDE < cut) left else right
    }

    private fun vBars(p: IntArray, n: Int, bar: Double, color: Int) {
        for (k in 0 until n) {
            val cx = (k + 0.5) / n
            fillRect(p, cx, 0.5, bar, 1.0, color)
        }
    }

    /** Filled band along an axis through the centre; angle in degrees. */
    private fun band(p: IntArray, angleDeg: Double, thickness: Double, color: Int) {
        val rad = Math.toRadians(angleDeg)
        val nx = kotlin.math.cos(rad); val ny = kotlin.math.sin(rad)
        val half = thickness / 2.0
        for (y in 0 until SIDE) for (x in 0 until SIDE) {
            val dx = (x + 0.5) / SIDE - 0.5
            val dy = (y + 0.5) / SIDE - 0.5
            if (abs(dx * nx + dy * ny) <= half) p[y * SIDE + x] = color
        }
    }

    private fun segment(p: IntArray, ax: Double, ay: Double, bx: Double, by: Double,
                        thickness: Double, color: Int) {
        val axp = ax * SIDE; val ayp = ay * SIDE; val bxp = bx * SIDE; val byp = by * SIDE
        val vx = bxp - axp; val vy = byp - ayp
        val len2 = vx * vx + vy * vy
        val halfPx = thickness * SIDE / 2.0
        for (y in 0 until SIDE) for (x in 0 until SIDE) {
            val t = (((x + 0.5 - axp) * vx + (y + 0.5 - ayp) * vy) / len2).coerceIn(0.0, 1.0)
            val px = axp + t * vx; val py = ayp + t * vy
            val d = sqrt((x + 0.5 - px) * (x + 0.5 - px) + (y + 0.5 - py) * (y + 0.5 - py))
            if (d <= halfPx) p[y * SIDE + x] = color
        }
    }

    /** Re-render a single-blob scene from parameters (closed-loop simulation). */
    fun blobPixels(b: Blob): IntArray {
        val p = canvas(b.ground)
        fillRect(p, b.cx, b.cy, b.size, b.size, b.content)
        return p
    }

    /** Fixed scene order: flat baseline, the position/size scan, then the texture families. */
    val all: List<Scene> = buildList {
        add(Scene("flat", "flat", "featureless ground: conservative fallback expected",
            canvas(MID)))

        // The scan: 3 blob sizes × 7×7 interior grid positions, row/column contiguous so
        // adjacent-list edges are small camera moves.
        for (size in listOf(0.10, 0.22, 0.38)) {
            val sizeTag = when (size) { 0.10 -> "10"; 0.22 -> "22"; else -> "38" }
            for (j in 1..7) for (i in 1..7) {
                val cx = i / 8.0; val cy = j / 8.0
                val b = Blob(cx, cy, size)
                add(Scene("blob_${sizeTag}_${j}_$i",
                    "blob_grid", "single mass, grid scan", blobPixels(b), b))
            }
        }

        // Small mass in a quiet field (negative space / minimal), also closed-loop eligible.
        for ((k, pos) in listOf(1.0 / 3 to 1.0 / 3, 2.0 / 3 to 1.0 / 3,
            1.0 / 3 to 2.0 / 3, 0.5 to 0.62).withIndex()) {
            val b = Blob(pos.first, pos.second, 0.07, PALE_GROUND, DARK)
            add(Scene("minimal_$k", "minimal", "single small mass in a pale quiet field",
                blobPixels(b), b))
        }

        // Content crossing the frame edges: 4 edge centres + 4 corners, two sizes.
        val edgeSpots = listOf(
            0.0 to 0.5, 1.0 to 0.5, 0.5 to 0.0, 0.5 to 1.0,
            0.0 to 0.0, 1.0 to 0.0, 0.0 to 1.0, 1.0 to 1.0,
        )
        for ((k, spot) in edgeSpots.withIndex()) for (size in listOf(0.18, 0.30)) {
            val b = Blob(spot.first, spot.second, size)
            add(Scene("edge_${k}_$size", "edge", "mass centred on a frame edge/corner",
                blobPixels(b)))
        }

        // Busy borders with a non-empty middle: the too-tight geometry.
        run {
            val p = canvas(DARK)
            for (k in 0..4) {
                val t = (k + 0.5) / 5
                fillRect(p, t, 0.035, 0.07, 0.07, LIGHT)
                fillRect(p, t, 0.965, 0.07, 0.07, LIGHT)
                fillRect(p, 0.035, t, 0.07, 0.07, LIGHT)
                fillRect(p, 0.965, t, 0.07, 0.07, LIGHT)
            }
            fillRect(p, 0.5, 0.5, 0.18, 0.18, LIGHT)
            add(Scene("crowd_all_edges", "crowd", "dense masses on all four edges + centre", p))
        }
        run {
            val p = canvas(DARK)
            for (k in 0..4) {
                val t = (k + 0.5) / 5
                fillRect(p, t, 0.035, 0.07, 0.07, LIGHT)
                fillRect(p, 0.035, t, 0.07, 0.07, LIGHT)
                fillRect(p, 0.965, t, 0.07, 0.07, LIGHT)
            }
            fillRect(p, 0.5, 0.5, 0.16, 0.16, LIGHT)
            add(Scene("crowd_three_edges", "crowd", "dense masses on three edges + centre", p))
        }

        // Horizon walking from top to bottom in 5% steps; mid band is the bisected frame.
        for (k in 3..17) {
            val y = k / 20.0
            val p = canvas(MID); hSplit(p, y, UPPER, LOWER)
            add(Scene("hline_$k", "hline", "single horizontal divider at y=$y", p))
        }

        for (n in listOf(3, 5, 7, 9)) {
            val p = canvas(DARK); vBars(p, n, 0.045, LIGHT)
            add(Scene("vbars_$n", "vbars", "$n evenly spaced vertical bars", p))
        }

        add(Scene("diag_slash", "diag", "45° light band",
            canvas(DARK).also { band(it, 45.0, 0.07, LIGHT) }))
        add(Scene("diag_backslash", "diag", "−45° light band",
            canvas(DARK).also { band(it, -45.0, 0.07, LIGHT) }))

        run {
            val p = canvas(DARK)
            segment(p, 0.02, 0.02, 0.5, 0.5, 0.02, LIGHT)
            segment(p, 0.98, 0.02, 0.5, 0.5, 0.02, LIGHT)
            segment(p, 0.02, 0.98, 0.5, 0.5, 0.02, LIGHT)
            segment(p, 0.98, 0.98, 0.5, 0.5, 0.02, LIGHT)
            add(Scene("converge", "conv", "four corner-to-centre converging lines", p))
        }

        add(Scene("sym_lr", "symmetry", "mirrored pair left/right",
            canvas(DARK).also {
                fillRect(it, 0.30, 0.5, 0.20, 0.20, LIGHT)
                fillRect(it, 0.70, 0.5, 0.20, 0.20, LIGHT)
            }))
        add(Scene("sym_tb", "symmetry", "mirrored pair top/bottom",
            canvas(DARK).also {
                fillRect(it, 0.5, 0.30, 0.20, 0.20, LIGHT)
                fillRect(it, 0.5, 0.70, 0.20, 0.20, LIGHT)
            }))
        add(Scene("sym_columns", "symmetry", "tight mirrored column pair",
            canvas(DARK).also {
                fillRect(it, 0.42, 0.5, 0.10, 0.66, LIGHT)
                fillRect(it, 0.58, 0.5, 0.10, 0.66, LIGHT)
            }))
        add(Scene("sym_wide", "symmetry", "wide mirrored pair",
            canvas(DARK).also {
                fillRect(it, 0.25, 0.5, 0.16, 0.16, LIGHT)
                fillRect(it, 0.75, 0.5, 0.16, 0.16, LIGHT)
            }))

        add(Scene("tri_up", "triangle", "three masses in an up triangle",
            canvas(DARK).also {
                fillRect(it, 0.5, 0.28, 0.13, 0.13, LIGHT)
                fillRect(it, 0.30, 0.68, 0.13, 0.13, LIGHT)
                fillRect(it, 0.70, 0.68, 0.13, 0.13, LIGHT)
            }))
        add(Scene("tri_down", "triangle", "three masses in a down triangle",
            canvas(DARK).also {
                fillRect(it, 0.30, 0.30, 0.13, 0.13, LIGHT)
                fillRect(it, 0.70, 0.30, 0.13, 0.13, LIGHT)
                fillRect(it, 0.5, 0.72, 0.13, 0.13, LIGHT)
            }))

        run {
            val p = canvas(MID); vSplit(p, 0.5, HALF_DARK, HALF_LIGHT)
            add(Scene("half_v", "half", "vertical half-dark / half-light split", p))
        }
        run {
            val p = canvas(MID); hSplit(p, 0.5, HALF_DARK, HALF_LIGHT)
            add(Scene("half_h", "half", "horizontal half-dark / half-light split", p))
        }

        for ((k, t) in listOf(0.03, 0.07).withIndex()) {
            val p = canvas(DARK)
            fillRect(p, 0.06, 0.5, t, 0.88, LIGHT)
            fillRect(p, 0.94, 0.5, t, 0.88, LIGHT)
            fillRect(p, 0.5, 0.06, 0.88, t, LIGHT)
            fillRect(p, 0.5, 0.94, 0.88, t, LIGHT)
            add(Scene("frame_$k", "frame_rect", "rectangular picture frame, thickness $t", p))
        }

        add(Scene("depth_1", "depth", "foreground band + small distant mass",
            canvas(DARK).also {
                fillRect(it, 0.5, 0.90, 1.0, 0.20, LIGHT)
                fillRect(it, 0.5, 0.30, 0.12, 0.12, LIGHT)
            }))
        add(Scene("depth_2", "depth", "lower foreground band + off-centre mass",
            canvas(DARK).also {
                fillRect(it, 0.30, 0.90, 0.60, 0.20, LIGHT)
                fillRect(it, 0.32, 0.34, 0.12, 0.12, LIGHT)
            }))
    }
}

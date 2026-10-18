package dev.streetsnap.camera

/**
 * Verbatim extraction from street-snap camera/.../CompositionPipelineContracts.kt
 * (street-snap@e891092, 2026-10-19). Only this class is copied: the rest of the source
 * file references CompositionFrame, whose definition lives next to CameraX bindings and
 * cannot be part of a JVM-only harness. The class body below is unchanged.
 */

/** A rectangle in the uncropped camera frame, with all coordinates normalized to 0..1. */
data class NormalizedBoundingBox(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
) {
    init {
        require(left.isFinite() && top.isFinite() && right.isFinite() && bottom.isFinite())
        require(left in 0f..1f && top in 0f..1f && right in 0f..1f && bottom in 0f..1f)
        require(left < right && top < bottom)
    }

    fun interpolate(target: NormalizedBoundingBox, amount: Float): NormalizedBoundingBox {
        val t = amount.coerceIn(0f, 1f)
        return NormalizedBoundingBox(
            left + (target.left - left) * t,
            top + (target.top - top) * t,
            right + (target.right - right) * t,
            bottom + (target.bottom - bottom) * t,
        )
    }
}

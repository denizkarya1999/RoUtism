// File: TemplateMatcher.kt
package com.developer27.xemotion.secondaryprocessing

import android.content.res.AssetManager
import android.graphics.BitmapFactory
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.io.IOException

/**
 * Loads reference overlays and extracts feature vectors for ML input
 */
class TemplateMatcher(private val assetManager: AssetManager) {
    private val CLASSES = listOf("Angry", "Anxious", "Excitement", "Sadness")
    private val trainTemplates: Map<String, List<Mat>> = loadTemplates("dataset/train")

    fun loadTemplates(root: String): Map<String, List<Mat>> {
        return CLASSES.associateWith { cls ->
            (assetManager.list("$root/$cls") ?: emptyArray())
                .filter { it.endsWith(".jpg", ignoreCase = true) }
                .mapNotNull { fname ->
                    try {
                        assetManager.open("$root/$cls/$fname").use { stream ->
                            BitmapFactory.decodeStream(stream).let { bmp ->
                                Mat().apply { Utils.bitmapToMat(bmp, this) }.apply {
                                    Imgproc.cvtColor(this, this, Imgproc.COLOR_RGBA2GRAY)
                                    Imgproc.resize(this, this, org.opencv.core.Size(79.0, 68.0))
                                }
                            }
                        }
                    } catch (_: IOException) {
                        null
                    }
                }
        }
    }

    /**
     * Computes a feature vector where each entry is the minimal pixel-difference
     * between the live trace and all class templates (k=1 nearest neighbor).
     */
    fun extractFeatures(trace: Mat): FloatArray {
        return CLASSES.map { cls ->
            val min = trainTemplates[cls]?.minOfOrNull { ref ->
                val diff = Mat()
                Core.absdiff(trace, ref, diff)
                val cnt = Core.countNonZero(diff).toDouble().also { diff.release() }
                cnt
            } ?: Double.MAX_VALUE
            min.toFloat()
        }.toFloatArray()
    }
}

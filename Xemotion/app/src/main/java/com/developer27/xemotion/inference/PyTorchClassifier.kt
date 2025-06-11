// File: PyTorchClassifier3.kt
package com.developer27.xemotion.inference

import android.content.Context
import android.graphics.Bitmap
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import kotlin.math.exp

/**
 * A PyTorch classifier that slices a 4-logit model into 3 labels.
 */
class PyTorchClassifier private constructor(
    private val module: Module,
    val inputWidth: Int = 224,
    val inputHeight: Int = 224
) : AutoCloseable {

    companion object {
        @Volatile private var cachedModule: Module? = null

        fun fromAsset(
            context: Context,
            modelAsset: String = "resnet50_emotion.pt",
            inputWidth: Int = 224,
            inputHeight: Int = 224
        ): PyTorchClassifier {
            val mod = cachedModule ?: synchronized(this) {
                cachedModule ?: PyTorchModuleLoader
                    .loadModule(context, modelAsset)
                    .also { cachedModule = it }
            }
            return PyTorchClassifier(mod, inputWidth, inputHeight)
        }
    }

    /** The 3 labels you want to show */
    private val labels = listOf("Sadness", "Excitement", "Anxious")

    // ImageNet normalization constants
    private val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val std  = floatArrayOf(0.229f, 0.224f, 0.225f)

    /** Reusable softmax buffer of size 3 */
    private val softmaxBuffer = FloatArray(labels.size)

    /**
     * Classify a single bitmap:
     * 1) run the 4-logit model
     * 2) slice out [Sadness=idx2, Excitement=idx1, Anxious=idx0]
     * 3) softmax the 3-element slice
     */
    fun classifyLine(bitmap: Bitmap): Pair<String, FloatArray> {
        val scaled = Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true)
        val tensor = TensorImageUtils.bitmapToFloat32Tensor(scaled, mean, std)

        // 1) get 4 logits
        val logits4 = module.forward(IValue.from(tensor)).toTensor().dataAsFloatArray
        // 2) pick exactly the 3 we care about (in labels order)
        val logits3 = floatArrayOf(
            logits4[0],  // Anxious
            logits4[1],  // Excitement
            logits4[2]   // Sadness
        )
        // 3) softmax
        val probs = softmax(logits3)
        // 4) best index
        val best  = probs.indices.maxByOrNull { probs[it] } ?: 0
        return labels[best] to probs
    }

    private fun softmax(logits: FloatArray): FloatArray {
        var maxLogit = logits[0]
        for (i in 1 until logits.size) if (logits[i] > maxLogit) maxLogit = logits[i]
        var sum = 0f
        for (i in logits.indices) {
            val e = exp(logits[i] - maxLogit)
            softmaxBuffer[i] = e
            sum += e
        }
        for (i in softmaxBuffer.indices) {
            softmaxBuffer[i] = softmaxBuffer[i] / sum
        }
        return softmaxBuffer
    }

    override fun close() {
        module.destroy()
    }
}
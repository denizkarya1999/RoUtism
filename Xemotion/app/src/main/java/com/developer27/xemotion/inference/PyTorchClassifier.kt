// File: PyTorchClassifier.kt
package com.developer27.xemotion.inference

import android.content.Context
import android.graphics.Bitmap
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils

/**
 * A PyTorch classifier that loads a TorchScript model and labels from assets.
 */
class PyTorchClassifier(
    context: Context,
    private val modelAsset: String = "resnet50_emotion.pt",
    private val labelAsset: String = "labels.txt",
    val inputWidth: Int = 224,
    val inputHeight: Int = 224
) : AutoCloseable {

    private val module: Module =
        PyTorchModuleLoader.loadModule(context, modelAsset)

    /** Public labels list for your Activity to read counts & summaries */
    val labels: List<String> = context.assets.open(labelAsset)
        .bufferedReader()
        .useLines { it.toList() }

    // ImageNet normalization constants
    private val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val std  = floatArrayOf(0.229f, 0.224f, 0.225f)

    /** Classify a single bitmap. Returns (bestLabel, allProbabilities). */
    fun classifyLine(bitmap: Bitmap): Pair<String, FloatArray> {
        val scaled = Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true)
        val tensor = TensorImageUtils.bitmapToFloat32Tensor(scaled, mean, std)
        val output = module.forward(IValue.from(tensor)).toTensor()
        val scores = output.dataAsFloatArray
        val probs  = softmax(scores)
        val best   = probs.indices.maxByOrNull { probs[it] } ?: 0
        return labels[best] to probs
    }

    /** Classify a batch of bitmaps. */
    fun classifyBatch(bitmaps: List<Bitmap>): List<Pair<String, FloatArray>> =
        bitmaps.map { classifyLine(it) }

    /** Simple softmax over logits. */
    private fun softmax(logits: FloatArray): FloatArray {
        val max = logits.maxOrNull() ?: 0f
        val exps = logits.map { Math.exp((it - max).toDouble()) }
        val sum  = exps.sum()
        return exps.map { (it / sum).toFloat() }.toFloatArray()
    }

    /** Release the native module when done. */
    override fun close() {
        module.destroy()
    }
}
package com.developer27.xemotion.inference

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import java.io.Closeable

class PyTorchClassifier private constructor(
    private val module: Module,
    private val inputWidth: Int = 224,
    private val inputHeight: Int = 224,
    private val mean: FloatArray = floatArrayOf(0.485f, 0.456f, 0.406f),
    private val std: FloatArray = floatArrayOf(0.229f, 0.224f, 0.225f)
) : Closeable {

    companion object {
        private const val TAG = "PyTorchClassifier"
        @Volatile private var instance: PyTorchClassifier? = null

        // Hardcoded emotion labels
        private val labels = listOf("Angry","Anxious", "Disgusted", "Excited", "Sad")

        /**
         * Load the .pt model from assets and return (or cache) the classifier.
         * Call: PyTorchClassifier.fromAsset(context, "resnet50_emotion.pt")
         */
        fun fromAsset(
            context: Context,
            modelAsset: String = "resnet50_emotion.pt"
        ): PyTorchClassifier {
            return instance ?: synchronized(this) {
                val module = PyTorchModuleLoader.loadModule(context, modelAsset)
                PyTorchClassifier(module).also { instance = it }
            }
        }
    }

    /**
     * Runs the model on the given bitmap and returns:
     *  • bestLabel: highest probability class (after softmax)
     *  • probs: FloatArray of softmax probabilities
     */
    fun classifyLine(bitmap: Bitmap): Pair<String, FloatArray> {
        // 1) Preprocess: resize & normalize
        val scaled = Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true)
        val tensor = TensorImageUtils.bitmapToFloat32Tensor(scaled, mean, std)

        // 2) Forward pass
        val outputTensor = module.forward(IValue.from(tensor)).toTensor()
        val rawLogits = outputTensor.dataAsFloatArray

        // 3) Softmax
        val expScores = rawLogits.map { Math.exp(it.toDouble()).toFloat() }
        val sumExp = expScores.sum()
        val probs = expScores.map { it / sumExp }.toFloatArray()

        // 4) Argmax on probabilities
        val maxIndex = probs.indices.maxByOrNull { probs[it] } ?: 0
        val bestLabel = labels.getOrElse(maxIndex) { "Unknown" }

        return bestLabel to probs
    }

    /**
     * Same as classifyLine, but logs each label:probability for debugging.
     */
    fun classifyAndLog(bitmap: Bitmap) {
        val (_, probs) = classifyLine(bitmap)
        for (i in probs.indices) {
            Log.d(TAG, "${labels.getOrNull(i) ?: "Label$i"}: ${probs[i]}")
        }
    }

    override fun close() {
        module.destroy()
    }
}
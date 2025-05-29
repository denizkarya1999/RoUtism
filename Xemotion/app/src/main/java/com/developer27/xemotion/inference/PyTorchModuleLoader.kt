package com.developer27.xemotion.inference

import android.content.Context
import org.pytorch.Module
import java.io.File
import java.io.FileOutputStream

/**
 * Utility to load a .pt model bundled in assets into a PyTorch Mobile Module.
 */
object PyTorchModuleLoader {
    /**
     * Copies the asset <assetName> from assets to internal storage and loads it as a PyTorch Module.
     * @param context the Android context
     * @param assetName the filename of the .pt model in assets (e.g., "model.pt")
     * @return the loaded PyTorch Module
     */
    fun loadModule(context: Context, assetName: String): Module {
        val file = File(context.filesDir, assetName)
        if (!file.exists() || file.length() == 0L) {
            context.assets.open(assetName).use { input ->
                FileOutputStream(file).use { output ->
                    input.copyTo(output)
                }
            }
        }
        return Module.load(file.absolutePath)
    }
}
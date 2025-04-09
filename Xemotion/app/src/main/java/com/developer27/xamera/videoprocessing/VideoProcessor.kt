@file:Suppress("SameParameterValue")

package com.developer27.xemotion.videoprocessing

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.apache.commons.math3.analysis.interpolation.SplineInterpolator
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.video.KalmanFilter
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage
import java.util.LinkedList
import kotlin.math.max
import kotlin.math.min

data class DetectionResult(
    val xCenter: Float, val yCenter: Float,
    val width: Float, val height: Float,
    val confidence: Float
)
data class BoundingBox(
    val x1: Float, val y1: Float,
    val x2: Float, val y2: Float,
    val confidence: Float, val classId: Int
)

private var tfliteInterpreter: Interpreter? = null
private val rawDataList = LinkedList<Point>()
private val smoothDataList = LinkedList<Point>()

// Object to hold various configuration settings.
object Settings {
    object DetectionMode {
        enum class Mode { CONTOUR, YOLO }
        var current: Mode = Mode.YOLO
        var enableYOLOinference = true
    }
    object Inference {
        var confidenceThreshold: Float = 0.5f
        var iouThreshold: Float = 0.5f
    }
    object Trace {
        var enableRAWtrace = false
        var enableSPLINEtrace = true
        var lineLimit = 50
        var splineStep = 0.01
        var originalLineColor = Scalar(0.0, 39.0, 76.0)
        var splineLineColor = Scalar(255.0, 203.0, 5.0)
        var lineThickness = 4
    }
    object BoundingBox {
        var enableBoundingBox = true
        var boxColor = Scalar(0.0, 39.0, 76.0)
        var boxThickness = 2
    }
    object Brightness {
        var factor = 2.0
        var threshold = 150.0
    }
    object ExportData {
        var frameIMG = false
        var videoDATA = false
    }
}

// Main VideoProcessor class.
class VideoProcessor(private val context: Context) {

    init {
        initOpenCV()
        KalmanHelper.initKalmanFilter()
    }
    private fun initOpenCV() {
        try {
            System.loadLibrary("opencv_java4")
        } catch (e: UnsatisfiedLinkError) {
            Log.d("VideoProcessor","OpenCV failed to load: ${e.message}", e)
        }
    }
    fun setInterpreter(model: Interpreter) {
        synchronized(this) { tfliteInterpreter = model }
        Log.d("VideoProcessor","TFLite Model set in VideoProcessor successfully!")
    }
    fun reset() {
        rawDataList.clear()
        smoothDataList.clear()
    }

    // Processes a frame asynchronously and returns a Pair (outputBitmap, videoBitmap).
    fun processFrame(bitmap: Bitmap, callback: (Pair<Bitmap, Bitmap>?) -> Unit) {
        CoroutineScope(Dispatchers.Default).launch {
            val result: Pair<Bitmap, Bitmap>? = try {
                when (Settings.DetectionMode.current) {
                    Settings.DetectionMode.Mode.CONTOUR -> processFrameInternalCONTOUR(bitmap)
                    Settings.DetectionMode.Mode.YOLO -> processFrameInternalYOLO(bitmap)
                }
            } catch (e: Exception) {
                Log.d("VideoProcessor","Error processing frame: ${e.message}", e)
                null
            }
            withContext(Dispatchers.Main) { callback(result) }
        }
    }
    // Processes a frame using Contour Detection - Returns a Pair containing outputBitmap and videoBitmap.
    private fun processFrameInternalCONTOUR(bitmap: Bitmap): Pair<Bitmap, Bitmap>? {
        return try {
            val (pMat, pBmp) = Preprocessing.preprocessFrame(bitmap)
            val (center, cMat) = ContourDetection.processContourDetection(pMat)
            TraceRenderer.drawTrace(center, cMat)
            val outBmp = Bitmap.createBitmap(cMat.cols(), cMat.rows(), Bitmap.Config.ARGB_8888).also { Utils.matToBitmap(cMat, it) }
            pMat.release(); cMat.release()
            outBmp to pBmp
        } catch (e: Exception) {
            Log.d("VideoProcessor","Error processing frame: ${e.message}", e)
            null
        }
    }
    // Processes a frame using YOLO - Returns a Pair containing outputBitmap and letterboxedBitmap.
    private suspend fun processFrameInternalYOLO(bitmap: Bitmap): Pair<Bitmap, Bitmap> = withContext(Dispatchers.IO) {
        val (inputW, inputH, outputShape) = getModelDimensions()
        val (letterboxed, offsets) = YOLOHelper.createLetterboxedBitmap(bitmap, inputW, inputH)
        val m = Mat().also { Utils.bitmapToMat(bitmap, it) }
        if (Settings.DetectionMode.enableYOLOinference && tfliteInterpreter != null) {
            val out = Array(outputShape[0]) { Array(outputShape[1]) { FloatArray(outputShape[2]) } }
            TensorImage(DataType.FLOAT32).apply { load(letterboxed) }.also { tfliteInterpreter?.run(it.buffer, out) }
            YOLOHelper.parseTFLite(out)?.let {
                val (box, c) = YOLOHelper.rescaleInferencedCoordinates(it, bitmap.width, bitmap.height, offsets, inputW, inputH)
                if (Settings.BoundingBox.enableBoundingBox) YOLOHelper.drawBoundingBoxes(m, box)
                TraceRenderer.drawTrace(c, m)
            }
        }
        val yoloBmp = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888).also {
            Utils.matToBitmap(m, it)
            m.release()
        }
        yoloBmp to letterboxed
    }

    // Dynamically retrieves the model input size.
    fun getModelDimensions(): Triple<Int, Int, List<Int>> {
        val inTensor = tfliteInterpreter?.getInputTensor(0)
        val inShape = inTensor?.shape()
        val (h, w) = (inShape?.getOrNull(1) ?: 416) to (inShape?.getOrNull(2) ?: 416)
        val outTensor = tfliteInterpreter?.getOutputTensor(0)
        val outShape = outTensor?.shape()?.toList() ?: listOf(1, 5, 3549)
        return Triple(w, h, outShape)
    }
    // Creates a white, square (28x28) Bitmap that encapsulates the drawn spline trace (with padding).
    fun exportTraceForInference(): Bitmap {
        // Ensure there is some trace data.
        if (smoothDataList.isEmpty()) {
            return Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888).apply { eraseColor(Color.WHITE) }
        }
        // 1. Compute the bounding box of the trace points.
        var minX = Double.MAX_VALUE
        var minY = Double.MAX_VALUE
        var maxX = Double.MIN_VALUE
        var maxY = Double.MIN_VALUE
        for (pt in smoothDataList) {
            minX = min(minX, pt.x)
            minY = min(minY, pt.y)
            maxX = max(maxX, pt.x)
            maxY = max(maxY, pt.y)
        }
        // 2. Define padding (in pixels) around the output img.
        val padding = 30.0
        // Compute optimal dimensions.
        val (optimalWidth, optimalHeight) = (max((maxX - minX + 2 * padding).toInt(), 1)) to (max((maxY - minY + 2 * padding).toInt(), 1))
        // 3. Determine the square size as the greatest of the optimal dimensions.
        val squareSize = max(optimalWidth, optimalHeight)
        // 4. Create a white square Mat of the computed dimensions.
        val mat = Mat(squareSize, squareSize, CvType.CV_8UC4, Scalar(255.0, 255.0, 255.0, 255.0))
        // 5. Compute offsets to center the drawn trace inside the square.
        val (xOffset, yOffset) = ((squareSize - optimalWidth) / 2.0) to ((squareSize - optimalHeight) / 2.0)
        // 6. Create an adjusted list of points so that the drawing starts at (padding, padding) plus the offsets.
        val adjustedPoints = smoothDataList.map { Point(it.x - minX + padding + xOffset, it.y - minY + padding + yOffset) }
        // 7. Set up drawing parameters (temporarily override settings).
        val originalColor = Settings.Trace.splineLineColor
        val originalThickness = Settings.Trace.lineThickness
        Settings.Trace.splineLineColor = Scalar(0.0, 0.0, 0.0) // Black
        Settings.Trace.lineThickness = 40
        // 8. Draw the spline curve using the adjusted points.
        TraceRenderer.drawSplineCurve(adjustedPoints, mat)
        // 9. Restore the original settings.
        Settings.Trace.splineLineColor = originalColor
        Settings.Trace.lineThickness = originalThickness
        // 10. Convert the Mat back to a Bitmap.
        val outputBitmap = Bitmap.createBitmap(squareSize, squareSize, Bitmap.Config.ARGB_8888).apply {
            Utils.matToBitmap(mat, this)
            mat.release()
        }
        val scaledBitmap = Bitmap.createScaledBitmap(outputBitmap, 28, 28, true)
        return scaledBitmap
    }

// NEW FUNCTION: Creates a fixed-size (based on 80% of the device's screen size) white, square Bitmap
// that encapsulates the drawn spline trace (with padding) and returns the full-size image
// for data collection. If the drawn trace exceeds the effective area, it is scaled down; otherwise, its original size is preserved.
    fun exportTraceForDataCollection(): Bitmap {
        // Ensure there is some trace data.
        if (smoothDataList.isEmpty()) {
            return Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888)
                .apply { eraseColor(Color.WHITE) }
        }

        // Obtain screen dimensions (choose the smaller dimension)
        val metrics = context.resources.displayMetrics
        val screenWidth = metrics.widthPixels
        val screenHeight = metrics.heightPixels
        // Use 75% of the smaller screen dimension for the fixed pane size.
        val fixedSize = (min(screenWidth, screenHeight) * 0.75).toDouble()

        // 1. Compute the bounding box of the trace points.
        var minX = Double.MAX_VALUE
        var minY = Double.MAX_VALUE
        var maxX = Double.MIN_VALUE
        var maxY = Double.MIN_VALUE
        for (pt in smoothDataList) {
            minX = min(minX, pt.x)
            minY = min(minY, pt.y)
            maxX = max(maxX, pt.x)
            maxY = max(maxY, pt.y)
        }

        // 2. Define padding (in pixels) around the drawing.
        val padding = 30.0

        // 3. Compute the drawing width and height.
        val drawingWidth = maxX - minX
        val drawingHeight = maxY - minY

        // 4. Determine the effective drawing area.
        val effectiveSize = fixedSize - 2 * padding

        // 5. Determine scaling factor:
        //    Scale down if the drawing exceeds the effective area; otherwise, leave at original scale.
        val scale = if (drawingWidth > effectiveSize || drawingHeight > effectiveSize) {
            min(effectiveSize / drawingWidth, effectiveSize / drawingHeight)
        } else {
            1.0
        }

        // 6. Compute new drawing dimensions after scaling.
        val newDrawingWidth = drawingWidth * scale
        val newDrawingHeight = drawingHeight * scale

        // 7. Compute offsets to center the drawing in the fixed pane.
        val offsetX = (fixedSize - newDrawingWidth) / 2.0
        val offsetY = (fixedSize - newDrawingHeight) / 2.0

        // 8. Create an adjusted list of points by scaling and translating the original points.
        val adjustedPoints = smoothDataList.map {
            Point((it.x - minX) * scale + offsetX, (it.y - minY) * scale + offsetY)
        }

        // 9. Create a fixed-size white square Mat.
        val mat = Mat(fixedSize.toInt(), fixedSize.toInt(), CvType.CV_8UC4, Scalar(255.0, 255.0, 255.0, 255.0))

        // 10. Set up drawing parameters (temporarily override settings).
        val originalColor = Settings.Trace.splineLineColor
        val originalThickness = Settings.Trace.lineThickness
        Settings.Trace.splineLineColor = Scalar(0.0, 0.0, 0.0) // Black
        Settings.Trace.lineThickness = 3  // As thin as possible
        // 11. Draw the spline curve using the adjusted points.
        TraceRenderer.drawSplineCurve(adjustedPoints, mat)
        // 12. Restore the original settings.
        Settings.Trace.splineLineColor = originalColor
        Settings.Trace.lineThickness = originalThickness
        // 13. Convert the Mat back to a Bitmap and return it.
        return Bitmap.createBitmap(fixedSize.toInt(), fixedSize.toInt(), Bitmap.Config.ARGB_8888).apply {
            Utils.matToBitmap(mat, this)
            mat.release()
        }
    }

    // Returns the tracking coordinates as a semicolon-separated string. Each point is formatted as "x,y,0.0".
    fun getTrackingCoordinatesString(): String {
        return smoothDataList.joinToString(separator = ";") { "${it.x},${it.y},0.0" }
    }
}

// Helper object to draw raw and spline traces.
object TraceRenderer {
    fun drawTrace(center: Point?, contourMat: Mat) {
        center?.let { detectedCenter ->
            rawDataList.add(detectedCenter)
            val (fx, fy) = KalmanHelper.applyKalmanFilter(detectedCenter)
            smoothDataList.add(Point(fx, fy))
        }
        with(Settings.Trace) {
            if (enableRAWtrace) drawRawTrace(rawDataList, contourMat)
            if (enableSPLINEtrace) drawSplineCurve(smoothDataList, contourMat)
        }
    }
    private fun drawRawTrace(data: List<Point>, image: Mat) {
        for (i in 1 until data.size) {
            Imgproc.line(image, data[i - 1], data[i], Settings.Trace.originalLineColor, Settings.Trace.lineThickness)
        }
    }
    fun drawSplineCurve(data: List<Point>, image: Mat) {
        if (data.size < 3) return
        val splinePair = applySplineInterpolation(data)
        val (splineX, splineY) = splinePair
        var prevPoint: Point? = null
        var t = 0.0
        val maxT = (data.size - 1).toDouble()
        while (t <= maxT) {
            val currentPoint = Point(splineX.value(t), splineY.value(t))
            prevPoint?.let { Imgproc.line(image, it, currentPoint, Settings.Trace.splineLineColor, Settings.Trace.lineThickness) }
            prevPoint = currentPoint
            t += Settings.Trace.splineStep
        }
    }
    private fun applySplineInterpolation(data: List<Point>): Pair<PolynomialSplineFunction, PolynomialSplineFunction> {
        val interpolator = SplineInterpolator()
        val xData = data.map { it.x }.toDoubleArray()
        val yData = data.map { it.y }.toDoubleArray()
        val tData = data.indices.map { it.toDouble() }.toDoubleArray()
        val splineX = interpolator.interpolate(tData, xData)
        val splineY = interpolator.interpolate(tData, yData)
        return Pair(splineX, splineY)
    }
}

// Helper object for applying a Kalman filter to smooth tracking points.
object KalmanHelper {
    private lateinit var kalmanFilter: KalmanFilter
    fun initKalmanFilter() {
        kalmanFilter = KalmanFilter(4, 2)
        kalmanFilter._transitionMatrix = Mat.eye(4, 4, CvType.CV_32F).apply {
            put(0, 2, 1.0)
            put(1, 3, 1.0)
        }
        kalmanFilter._measurementMatrix = Mat.eye(2, 4, CvType.CV_32F)
        kalmanFilter._processNoiseCov = Mat.eye(4, 4, CvType.CV_32F).apply { setTo(Scalar(1e-4)) }
        kalmanFilter._measurementNoiseCov = Mat.eye(2, 2, CvType.CV_32F).apply { setTo(Scalar(1e-2)) }
        kalmanFilter._errorCovPost = Mat.eye(4, 4, CvType.CV_32F)
    }
    fun applyKalmanFilter(point: Point): Pair<Double, Double> {
        val measurement = Mat(2, 1, CvType.CV_32F).apply {
            put(0, 0, point.x)
            put(1, 0, point.y)
        }
        kalmanFilter.predict()
        val corrected = kalmanFilter.correct(measurement)
        val fx = corrected[0, 0][0]
        val fy = corrected[1, 0][0]
        return fx to fy
    }
}

// Helper object for preprocessing frames with OpenCV.
object Preprocessing {
    fun preprocessFrame(src: Bitmap): Pair<Mat, Bitmap> {
        val sMat = Mat().also { Utils.bitmapToMat(src, it) }
        val gMat = Mat().also { Imgproc.cvtColor(sMat, it, Imgproc.COLOR_BGR2GRAY); sMat.release() }
        val eMat = Mat().also { Core.multiply(gMat, Scalar(Settings.Brightness.factor), it); gMat.release() }
        val tMat = Mat().also { Imgproc.threshold(eMat, it, Settings.Brightness.threshold, 255.0, Imgproc.THRESH_TOZERO); eMat.release() }
        val bMat = Mat().also { Imgproc.GaussianBlur(tMat, it, Size(5.0, 5.0), 0.0); tMat.release() }
        val k = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
        val cMat = Mat().also { Imgproc.morphologyEx(bMat, it, Imgproc.MORPH_CLOSE, k); bMat.release() }
        val bmp = Bitmap.createBitmap(cMat.cols(), cMat.rows(), Bitmap.Config.ARGB_8888).also { Utils.matToBitmap(cMat, it) }
        return cMat to bmp
    }
}

// Helper object for contour detection.
object ContourDetection {
    fun processContourDetection(mat: Mat) = findContours(mat).maxByOrNull { Imgproc.contourArea(it) }.let { c ->
        val center = c?.let {
            Imgproc.drawContours(mat, listOf(it), -1, Settings.BoundingBox.boxColor, Settings.BoundingBox.boxThickness)
            val m = Imgproc.moments(it)
            Point(m.m10 / m.m00, m.m01 / m.m00)
        }
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_GRAY2BGR)
        center to mat
    }
    private fun findContours(mat: Mat) = mutableListOf<MatOfPoint>().also {
        Mat().also { h -> Imgproc.findContours(mat, it, h, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE); h.release() }
    }
}

// Helper object for YOLO detection using TensorFlow Lite.
object YOLOHelper {
    fun parseTFLite(rawOutput: Array<Array<FloatArray>>): DetectionResult? {
        val numDetections = rawOutput[0][0].size
        // Step 1: Parse detections and filter by confidence.
        val detections = mutableListOf<DetectionResult>()
        for (i in 0 until numDetections) {
            val xCenter = rawOutput[0][0][i]
            val yCenter = rawOutput[0][1][i]
            val width = rawOutput[0][2][i]
            val height = rawOutput[0][3][i]
            val confidence = rawOutput[0][4][i]
            if (confidence >= Settings.Inference.confidenceThreshold) {
                detections.add(DetectionResult(xCenter, yCenter, width, height, confidence))
            }
        }
        if (detections.isEmpty()) {
            Log.d("YOLOTest", "No detections above confidence threshold: ${Settings.Inference.confidenceThreshold}")
            return null
        }
        // Step 2: Convert detections to bounding boxes.
        val detectionBoxes = detections.map { it to detectionToBox(it) }.toMutableList()
        // Sort by confidence (highest first).
        detectionBoxes.sortByDescending { it.first.confidence }
        // Step 3: Apply NMS.
        val nmsDetections = mutableListOf<DetectionResult>()
        while (detectionBoxes.isNotEmpty()) {
            val current = detectionBoxes.removeAt(0)
            nmsDetections.add(current.first)
            detectionBoxes.removeAll { other ->
                computeIoU(current.second, other.second) > Settings.Inference.iouThreshold
            }
        }
        // Step 4: Choose the detection with the highest confidence from the remaining.
        val bestDetection = nmsDetections.maxByOrNull { it.confidence }
        bestDetection?.let { d ->
            Log.d(
                "YOLOTest",
                "BEST DETECTION: confidence=${"%.8f".format(d.confidence)}, x_center=${d.xCenter}, y_center=${d.yCenter}, width=${d.width}, height=${d.height}"
            )
        }
        return bestDetection
    }
    private fun detectionToBox(d: DetectionResult) = BoundingBox(
        d.xCenter - d.width / 2,
        d.yCenter - d.height / 2,
        d.xCenter + d.width / 2,
        d.yCenter + d.height / 2,
        d.confidence,
        1
    )
    private fun computeIoU(boxA: BoundingBox, boxB: BoundingBox): Float {
        val x1 = max(boxA.x1, boxB.x1)
        val y1 = max(boxA.y1, boxB.y1)
        val x2 = min(boxA.x2, boxB.x2)
        val y2 = min(boxA.y2, boxB.y2)
        val intersectionWidth = max(0f, x2 - x1)
        val intersectionHeight = max(0f, y2 - y1)
        val intersectionArea = intersectionWidth * intersectionHeight
        val areaA = (boxA.x2 - boxA.x1) * (boxA.y2 - boxA.y1)
        val areaB = (boxB.x2 - boxB.x1) * (boxB.y2 - boxB.y1)
        val unionArea = areaA + areaB - intersectionArea
        return if (unionArea > 0f) intersectionArea / unionArea else 0f
    }
    fun rescaleInferencedCoordinates(detection: DetectionResult, originalWidth: Int, originalHeight: Int, padOffsets: Pair<Int, Int>, modelInputWidth: Int, modelInputHeight: Int): Pair<BoundingBox, Point> {
        // Compute the scale factor used in the letterbox transformation.
        val scale = min(modelInputWidth / originalWidth.toDouble(), modelInputHeight / originalHeight.toDouble())
        // Get the padding applied during letterboxing.
        val padLeft = padOffsets.first.toDouble()
        val padTop = padOffsets.second.toDouble()
        // Convert normalized coordinates to letterboxed image coordinates.
        val xCenterLetterboxed = detection.xCenter * modelInputWidth
        val yCenterLetterboxed = detection.yCenter * modelInputHeight
        val boxWidthLetterboxed = detection.width * modelInputWidth
        val boxHeightLetterboxed = detection.height * modelInputHeight
        // Remove padding and rescale back to original image coordinates.
        val xCenterOriginal = (xCenterLetterboxed - padLeft) / scale
        val yCenterOriginal = (yCenterLetterboxed - padTop) / scale
        val boxWidthOriginal = boxWidthLetterboxed / scale
        val boxHeightOriginal = boxHeightLetterboxed / scale
        // Compute bounding box corners in original image coordinates.
        val x1Original = xCenterOriginal - (boxWidthOriginal / 2)
        val y1Original = yCenterOriginal - (boxHeightOriginal / 2)
        val x2Original = xCenterOriginal + (boxWidthOriginal / 2)
        val y2Original = yCenterOriginal + (boxHeightOriginal / 2)
        Log.d("YOLOTest", "Adjusted BOUNDING BOX: x1=${"%.8f".format(x1Original)}, y1=${"%.8f".format(y1Original)}, x2=${"%.8f".format(x2Original)}, y2=${"%.8f".format(y2Original)}")
        // Create the bounding box and center point objects.
        val boundingBox = BoundingBox(
            x1Original.toFloat(),
            y1Original.toFloat(),
            x2Original.toFloat(),
            y2Original.toFloat(),
            detection.confidence,
            1 // Class index (or whatever label you're using)
        )
        val center = Point(xCenterOriginal, yCenterOriginal)
        return Pair(boundingBox, center)
    }
    fun drawBoundingBoxes(mat: Mat, box: BoundingBox) {
        val topLeft = Point(box.x1.toDouble(), box.y1.toDouble())
        val bottomRight = Point(box.x2.toDouble(), box.y2.toDouble())
        Imgproc.rectangle(mat, topLeft, bottomRight, Settings.BoundingBox.boxColor, Settings.BoundingBox.boxThickness)
        val label = "User_1 (${("%.2f".format(box.confidence * 100))}%)"
        val fontScale = 0.6
        val thickness = 1
        val baseline = IntArray(1)
        val textSize = Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX, fontScale, thickness, baseline)
        val textX = box.x1.toInt()
        val textY = (box.y1 - 5).toInt().coerceAtLeast(10)
        Imgproc.rectangle(
            mat,
            Point(textX.toDouble(), textY.toDouble() + baseline[0]),
            Point(textX + textSize.width, textY - textSize.height),
            Settings.BoundingBox.boxColor,
            Imgproc.FILLED
        )
        Imgproc.putText(
            mat,
            label,
            Point(textX.toDouble(), textY.toDouble()),
            Imgproc.FONT_HERSHEY_SIMPLEX,
            fontScale,
            Scalar(255.0, 255.0, 255.0),
            thickness
        )
    }
    fun createLetterboxedBitmap(srcBitmap: Bitmap, targetWidth: Int, targetHeight: Int, padColor: Scalar = Scalar(0.0, 0.0, 0.0)): Pair<Bitmap, Pair<Int, Int>> {
        val srcMat = Mat().also { Utils.bitmapToMat(srcBitmap, it) }
        val (srcWidth, srcHeight) = (srcMat.cols().toDouble()) to (srcMat.rows().toDouble())
        // Compute scaling factor: use the smaller ratio
        val scale = min(targetWidth / srcWidth, targetHeight / srcHeight)
        val (newWidth, newHeight) = (srcWidth * scale).toInt() to (srcHeight * scale).toInt()
        // Resize the source image
        val resized = Mat().also { Imgproc.resize(srcMat, it, Size(newWidth.toDouble(), newHeight.toDouble())) }
        srcMat.release()
        // Compute padding needed to reach target dimensions
        val (padWidth, padHeight) = (targetWidth - newWidth) to (targetHeight - newHeight)
        val computePadding = { total: Int -> total / 2 to (total - total / 2) }
        val (top, bottom) = computePadding(padHeight)
        val (left, right) = computePadding(padWidth)
        // Create the final letterboxed image with padding
        val letterboxed = Mat().also { Core.copyMakeBorder(resized, it, top, bottom, left, right, Core.BORDER_CONSTANT, padColor) }
        resized.release()
        // Convert the letterboxed Mat back to a Bitmap.
        val outputBitmap = Bitmap.createBitmap(letterboxed.cols(), letterboxed.rows(), srcBitmap.config).apply {
            Utils.matToBitmap(letterboxed, this)
            letterboxed.release()
        }
        // Return the letterboxed image and the top-left padding offset.
        return Pair(outputBitmap, Pair(left, top))
    }
}

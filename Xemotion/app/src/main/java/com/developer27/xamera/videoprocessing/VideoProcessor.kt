@file:Suppress("SameParameterValue")

package com.developer27.xemotion.videoprocessing

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Rect
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

// --------------------------------------------------
// Data Classes
// --------------------------------------------------
data class DetectionResult(
    val xCenter: Float,
    val yCenter: Float,
    val width: Float,
    val height: Float,
    val confidence: Float
)

data class BoundingBox(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val confidence: Float,
    val classId: Int
)

// --------------------------------------------------
// Globals
// --------------------------------------------------
private var tfliteInterpreter: Interpreter? = null
private val rawDataList = LinkedList<Point>()
private val smoothDataList = LinkedList<Point>()

// --------------------------------------------------
// Settings
// --------------------------------------------------
object Settings {
    object DetectionMode {
        enum class Mode { CONTOUR, YOLO }
        var current: Mode = Mode.CONTOUR
        var enableYOLOinference = false
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

        // Very thick trace lines
        var originalLineColor = Scalar(0.0, 39.0, 76.0)
        var splineLineColor = Scalar(255.0, 203.0, 5.0)
        var lineThickness = 100
    }

    object BoundingBox {
        var enableBoundingBox = true

        // Potentially thick bounding box for contours
        var boxColor = Scalar(0.0, 39.0, 76.0)
        var boxThickness = 50
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

// --------------------------------------------------
// VideoProcessor
// --------------------------------------------------
class VideoProcessor(private val context: Context) {

    init {
        initOpenCV()
        KalmanHelper.initKalmanFilter()
    }

    private fun initOpenCV() {
        try {
            System.loadLibrary("opencv_java4")
        } catch (e: UnsatisfiedLinkError) {
            Log.d("VideoProcessor", "OpenCV failed to load: ${e.message}", e)
        }
    }

    fun setInterpreter(model: Interpreter) {
        synchronized(this) {
            tfliteInterpreter = model
        }
        Log.d("VideoProcessor", "TFLite Model set in VideoProcessor successfully!")
    }

    fun reset() {
        rawDataList.clear()
        smoothDataList.clear()
    }

    /**
     * Processes a frame (bitmap) asynchronously.
     * Returns (debugOverlayBitmap, croppedRegion).
     *
     *  - debugOverlay: thresholded + contour + lines (for visible debugging).
     *  - croppedRegion: bounding box from the original color/grayscale frame.
     */
    fun processFrame(
        bitmap: Bitmap,
        callback: (Pair<Bitmap, Bitmap>?) -> Unit
    ) {
        CoroutineScope(Dispatchers.Default).launch {
            val result: Pair<Bitmap, Bitmap>? = try {
                when (Settings.DetectionMode.current) {
                    Settings.DetectionMode.Mode.CONTOUR -> processFrameInternalCONTOUR(bitmap)
                    Settings.DetectionMode.Mode.YOLO    -> processFrameInternalYOLO(bitmap)
                }
            } catch (e: Exception) {
                Log.d("VideoProcessor", "Error processing frame: ${e.message}", e)
                null
            }
            withContext(Dispatchers.Main) {
                callback(result)
            }
        }
    }

    /**
     * 1) Threshold & blur to find contours.
     * 2) Draw lines on cMat for debugging.
     * 3) Create a debug overlay (white blob + lines).
     * 4) Also crop from the original frame if boundingRect is found.
     * 5) Return the debug overlay as the FIRST image (so you see lines),
     *    and the cropped region as the SECOND image.
     */
    private fun processFrameInternalCONTOUR(bitmap: Bitmap): Pair<Bitmap, Bitmap>? {
        return try {
            // Step A: Preprocess => (pMat = thresholded Mat, pBmp = debug grayscale)
            val (pMat, _) = Preprocessing.preprocessFrame(bitmap)

            // Step B: Contour detection => (center, boundingRect, cMat)
            val (center, boundingRect, cMat) = ContourDetection.processContourDetection(pMat)

            // Draw lines on cMat
            TraceRenderer.drawTrace(center, cMat)

            // Convert cMat to a "debug overlay" => shows white blob & lines
            val debugOverlayBmp = Bitmap.createBitmap(cMat.cols(), cMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(cMat, debugOverlayBmp)

            // If boundingRect is found, crop from the original color frame
            val croppedRegion = if (boundingRect != null) {
                extractBoundingBoxRegion(bitmap, boundingRect)
            } else {
                // No contour => fallback to a 1x1 black or debugOverlay
                // We'll just fallback to debugOverlay here
                debugOverlayBmp
            }

            // Clean up
            pMat.release()
            cMat.release()

            // Return debugOverlay FIRST so you see lines on screen
            // and the cropped region as the second item
            return debugOverlayBmp to croppedRegion

        } catch (e: Exception) {
            Log.d("VideoProcessor", "Error in processFrameInternalCONTOUR: ${e.message}", e)
            null
        }
    }

    /**
     * YOLO-based detection (unchanged from prior code).
     */
    private suspend fun processFrameInternalYOLO(bitmap: Bitmap): Pair<Bitmap, Bitmap> =
        withContext(Dispatchers.IO) {
            val (inputW, inputH, outputShape) = getModelDimensions()
            val (letterboxed, offsets) = YOLOHelper.createLetterboxedBitmap(bitmap, inputW, inputH)

            val m = Mat().also { Utils.bitmapToMat(bitmap, it) }

            if (Settings.DetectionMode.enableYOLOinference && tfliteInterpreter != null) {
                val out = Array(outputShape[0]) {
                    Array(outputShape[1]) {
                        FloatArray(outputShape[2])
                    }
                }
                val inputBuffer = TensorImage(DataType.FLOAT32).apply { load(letterboxed) }
                tfliteInterpreter?.run(inputBuffer.buffer, out)

                YOLOHelper.parseTFLite(out)?.let { bestDetection ->
                    val (box, c) = YOLOHelper.rescaleInferencedCoordinates(
                        bestDetection,
                        bitmap.width,
                        bitmap.height,
                        offsets,
                        inputW,
                        inputH
                    )
                    if (Settings.BoundingBox.enableBoundingBox) {
                        YOLOHelper.drawBoundingBoxes(m, box)
                    }
                    TraceRenderer.drawTrace(c, m)
                }
            }

            val yoloBmp = Bitmap.createBitmap(
                bitmap.width,
                bitmap.height,
                Bitmap.Config.ARGB_8888
            ).also {
                Utils.matToBitmap(m, it)
                m.release()
            }
            // We'll pair the YOLO result (yoloBmp) with the letterboxed input
            yoloBmp to letterboxed
        }

    /**
     * Return model input & output shapes (for YOLO).
     */
    fun getModelDimensions(): Triple<Int, Int, List<Int>> {
        val inTensor = tfliteInterpreter?.getInputTensor(0)
        val inShape = inTensor?.shape()
        val (h, w) = (
                inShape?.getOrNull(1) ?: 416
                ) to (
                inShape?.getOrNull(2) ?: 416
                )
        val outTensor = tfliteInterpreter?.getOutputTensor(0)
        val outShape = outTensor?.shape()?.toList() ?: listOf(1, 5, 3549)

        return Triple(w, h, outShape)
    }

    /**
     * Crop boundingRect from the original color/grayscale frame.
     */
    private fun extractBoundingBoxRegion(srcBitmap: Bitmap, boundingRect: Rect): Bitmap {
        val left = boundingRect.left.coerceAtLeast(0)
        val top = boundingRect.top.coerceAtLeast(0)
        val right = boundingRect.right.coerceAtMost(srcBitmap.width)
        val bottom = boundingRect.bottom.coerceAtMost(srcBitmap.height)

        if (left >= right || top >= bottom) {
            // Invalid bounding box => fallback
            return Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888)
        }
        val width = right - left
        val height = bottom - top
        return Bitmap.createBitmap(srcBitmap, left, top, width, height)
    }

    /**
     * Exports a data-collection version of the *trace* (no bounding box logic).
     */
    fun exportTraceForDataCollection(): Bitmap {
        val snapshot = smoothDataList.toList()
        if (snapshot.isEmpty()) {
            return Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888).apply {
                eraseColor(Color.BLACK)
            }
        }

        // 1) bounding box among the smoothed points
        var minX = Double.MAX_VALUE
        var minY = Double.MAX_VALUE
        var maxX = Double.MIN_VALUE
        var maxY = Double.MIN_VALUE

        for (pt in snapshot) {
            minX = min(minX, pt.x)
            minY = min(minY, pt.y)
            maxX = max(maxX, pt.x)
            maxY = max(maxY, pt.y)
        }

        val width = (maxX - minX).coerceAtLeast(1.0)
        val height = (maxY - minY).coerceAtLeast(1.0)
        val padding = 30.0

        val matWidth = (width + 2 * padding).toInt().coerceAtLeast(1)
        val matHeight = (height + 2 * padding).toInt().coerceAtLeast(1)

        // 2) black Mat
        val mat = Mat(matHeight, matWidth, CvType.CV_8UC4, Scalar(0.0, 0.0, 0.0, 255.0))

        // 3) shift points by padding
        val adjustedPoints = snapshot.map {
            Point((it.x - minX) + padding, (it.y - minY) + padding)
        }

        // override color & thickness
        val originalColor = Settings.Trace.splineLineColor
        val originalThickness = Settings.Trace.lineThickness

        Settings.Trace.splineLineColor = Scalar(255.0, 255.0, 255.0)
        Settings.Trace.lineThickness = 10

        // 4) draw
        TraceRenderer.drawSplineCurve(adjustedPoints, mat)

        // restore color
        Settings.Trace.splineLineColor = originalColor
        Settings.Trace.lineThickness = originalThickness

        // 5) convert mat -> Bitmap
        val intermediateBitmap = Bitmap.createBitmap(matWidth, matHeight, Bitmap.Config.ARGB_8888).apply {
            Utils.matToBitmap(mat, this)
            mat.release()
        }

        // 6) scale to 79×68
        val finalWidth = 79
        val finalHeight = 68
        return Bitmap.createScaledBitmap(intermediateBitmap, finalWidth, finalHeight, true)
    }
}

// --------------------------------------------------
// TraceRenderer
// --------------------------------------------------
object TraceRenderer {
    fun drawTrace(center: Point?, contourMat: Mat) {
        center?.let { detectedCenter ->
            rawDataList.add(detectedCenter)
            val (fx, fy) = KalmanHelper.applyKalmanFilter(detectedCenter)
            smoothDataList.add(Point(fx, fy))
        }

        with(Settings.Trace) {
            if (enableRAWtrace) {
                drawRawTrace(rawDataList, contourMat)
            }
            if (enableSPLINEtrace) {
                drawSplineCurve(smoothDataList, contourMat)
            }
        }
    }

    private fun drawRawTrace(data: List<Point>, image: Mat) {
        for (i in 1 until data.size) {
            Imgproc.line(
                image,
                data[i - 1],
                data[i],
                Settings.Trace.originalLineColor,
                Settings.Trace.lineThickness
            )
        }
    }

    fun drawSplineCurve(data: List<Point>, image: Mat) {
        if (data.size < 3) return

        val (splineX, splineY) = applySplineInterpolation(data)
        var prevPoint: Point? = null
        var t = 0.0
        val maxT = (data.size - 1).toDouble()

        while (t <= maxT) {
            val currentPoint = Point(splineX.value(t), splineY.value(t))
            prevPoint?.let {
                Imgproc.line(
                    image,
                    it,
                    currentPoint,
                    Settings.Trace.splineLineColor,
                    Settings.Trace.lineThickness
                )
            }
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

// --------------------------------------------------
// KalmanHelper
// --------------------------------------------------
object KalmanHelper {
    private lateinit var kalmanFilter: KalmanFilter

    fun initKalmanFilter() {
        kalmanFilter = KalmanFilter(4, 2)
        kalmanFilter._transitionMatrix = Mat.eye(4, 4, CvType.CV_32F).apply {
            put(0, 2, 1.0)
            put(1, 3, 1.0)
        }
        kalmanFilter._measurementMatrix = Mat.eye(2, 4, CvType.CV_32F)
        kalmanFilter._processNoiseCov = Mat.eye(4, 4, CvType.CV_32F).apply {
            setTo(Scalar(1e-4))
        }
        kalmanFilter._measurementNoiseCov = Mat.eye(2, 2, CvType.CV_32F).apply {
            setTo(Scalar(1e-2))
        }
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

// --------------------------------------------------
// Preprocessing
// --------------------------------------------------
object Preprocessing {
    fun preprocessFrame(src: Bitmap): Pair<Mat, Bitmap> {
        val sMat = Mat().also { Utils.bitmapToMat(src, it) }

        val gMat = Mat().also {
            Imgproc.cvtColor(sMat, it, Imgproc.COLOR_BGR2GRAY)
            sMat.release()
        }
        val eMat = Mat().also {
            Core.multiply(gMat, Scalar(Settings.Brightness.factor), it)
            gMat.release()
        }
        val tMat = Mat().also {
            Imgproc.threshold(
                eMat,
                it,
                Settings.Brightness.threshold,
                255.0,
                Imgproc.THRESH_TOZERO
            )
            eMat.release()
        }
        val bMat = Mat().also {
            Imgproc.GaussianBlur(tMat, it, Size(5.0, 5.0), 0.0)
            tMat.release()
        }
        val k = Imgproc.getStructuringElement(
            Imgproc.MORPH_RECT,
            Size(3.0, 3.0)
        )
        val cMat = Mat().also {
            Imgproc.morphologyEx(bMat, it, Imgproc.MORPH_CLOSE, k)
            bMat.release()
        }
        val bmp = Bitmap.createBitmap(cMat.cols(), cMat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(cMat, bmp)

        return cMat to bmp
    }
}

// --------------------------------------------------
// ContourDetection
// --------------------------------------------------
object ContourDetection {

    /**
     * 1) Finds the largest contour.
     * 2) Draws it on the thresholded Mat.
     * 3) Computes bounding rectangle => returns an Android Rect.
     * 4) Uses image moments => returns (center, boundingRect, finalMat).
     */
    fun processContourDetection(mat: Mat): Triple<Point?, Rect?, Mat> {
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(
            mat,
            contours,
            hierarchy,
            Imgproc.RETR_EXTERNAL,
            Imgproc.CHAIN_APPROX_SIMPLE
        )
        hierarchy.release()

        if (contours.isEmpty()) {
            Imgproc.cvtColor(mat, mat, Imgproc.COLOR_GRAY2BGR)
            return Triple(null, null, mat)
        }

        val largestContour = contours.maxByOrNull { Imgproc.contourArea(it) }
            ?: run {
                Imgproc.cvtColor(mat, mat, Imgproc.COLOR_GRAY2BGR)
                return Triple(null, null, mat)
            }

        // Draw the largest contour
        Imgproc.drawContours(
            mat,
            listOf(largestContour),
            -1,
            Settings.BoundingBox.boxColor,
            Settings.BoundingBox.boxThickness
        )

        // Compute bounding rect
        val cvRect = Imgproc.boundingRect(largestContour)
        val boundingRect = Rect(
            cvRect.x,
            cvRect.y,
            cvRect.x + cvRect.width,
            cvRect.y + cvRect.height
        )

        if (Settings.BoundingBox.enableBoundingBox) {
            Imgproc.rectangle(
                mat,
                cvRect,
                Settings.BoundingBox.boxColor,
                Settings.BoundingBox.boxThickness
            )
        }

        val m = Imgproc.moments(largestContour)
        val center = Point(m.m10 / m.m00, m.m01 / m.m00)

        // Convert to BGR
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_GRAY2BGR)

        return Triple(center, boundingRect, mat)
    }
}

//------------------------------------------------
// YOLOHelper
// --------------------------------------------------
object YOLOHelper {

    fun parseTFLite(rawOutput: Array<Array<FloatArray>>): DetectionResult? {
        val numDetections = rawOutput[0][0].size
        val detections = mutableListOf<DetectionResult>()

        for (i in 0 until numDetections) {
            val xCenter = rawOutput[0][0][i]
            val yCenter = rawOutput[0][1][i]
            val width   = rawOutput[0][2][i]
            val height  = rawOutput[0][3][i]
            val confidence = rawOutput[0][4][i]

            if (confidence >= Settings.Inference.confidenceThreshold) {
                detections.add(
                    DetectionResult(xCenter, yCenter, width, height, confidence)
                )
            }
        }

        if (detections.isEmpty()) {
            Log.d("YOLOTest","No detections above threshold: ${Settings.Inference.confidenceThreshold}")
            return null
        }

        // Convert to boxes
        val detectionBoxes = detections.map { it to detectionToBox(it) }.toMutableList()
        detectionBoxes.sortByDescending { it.first.confidence }

        // Non-max suppression
        val nmsDetections = mutableListOf<DetectionResult>()
        while (detectionBoxes.isNotEmpty()) {
            val current = detectionBoxes.removeAt(0)
            nmsDetections.add(current.first)
            detectionBoxes.removeAll { other ->
                computeIoU(current.second, other.second) > Settings.Inference.iouThreshold
            }
        }

        val bestDetection = nmsDetections.maxByOrNull { it.confidence }
        bestDetection?.let { d ->
            Log.d(
                "YOLOTest",
                "BEST DETECTION: conf=${"%.2f".format(d.confidence)}, " +
                        "xC=${d.xCenter}, yC=${d.yCenter}, w=${d.width}, h=${d.height}"
            )
        }
        return bestDetection
    }

    private fun detectionToBox(d: DetectionResult) = BoundingBox(
        x1 = d.xCenter - d.width / 2,
        y1 = d.yCenter - d.height / 2,
        x2 = d.xCenter + d.width / 2,
        y2 = d.yCenter + d.height / 2,
        confidence = d.confidence,
        classId = 1
    )

    private fun computeIoU(boxA: BoundingBox, boxB: BoundingBox): Float {
        val x1 = max(boxA.x1, boxB.x1)
        val y1 = max(boxA.y1, boxB.y1)
        val x2 = min(boxA.x2, boxB.x2)
        val y2 = min(boxA.y2, boxB.y2)

        val intersectionW = max(0f, x2 - x1)
        val intersectionH = max(0f, y2 - y1)
        val intersectionArea = intersectionW * intersectionH

        val areaA = (boxA.x2 - boxA.x1) * (boxA.y2 - boxA.y1)
        val areaB = (boxB.x2 - boxB.x1) * (boxB.y2 - boxB.y1)
        val unionArea = areaA + areaB - intersectionArea

        return if (unionArea > 0f) intersectionArea / unionArea else 0f
    }

    fun rescaleInferencedCoordinates(
        detection: DetectionResult,
        originalWidth: Int,
        originalHeight: Int,
        padOffsets: Pair<Int, Int>,
        modelInputWidth: Int,
        modelInputHeight: Int
    ): Pair<BoundingBox, Point> {

        val scale = min(
            modelInputWidth / originalWidth.toDouble(),
            modelInputHeight / originalHeight.toDouble()
        )
        val (padLeft, padTop) = padOffsets.first.toDouble() to padOffsets.second.toDouble()

        val xCenterLetterboxed = detection.xCenter * modelInputWidth
        val yCenterLetterboxed = detection.yCenter * modelInputHeight

        val boxWidthLetterboxed  = detection.width  * modelInputWidth
        val boxHeightLetterboxed = detection.height * modelInputHeight

        val xCenterOriginal = (xCenterLetterboxed - padLeft) / scale
        val yCenterOriginal = (yCenterLetterboxed - padTop)  / scale

        val boxWidthOriginal  = boxWidthLetterboxed  / scale
        val boxHeightOriginal = boxHeightLetterboxed / scale

        val x1Original = xCenterOriginal - (boxWidthOriginal / 2)
        val y1Original = yCenterOriginal - (boxHeightOriginal / 2)
        val x2Original = xCenterOriginal + (boxWidthOriginal / 2)
        val y2Original = yCenterOriginal + (boxHeightOriginal / 2)

        val boundingBox = BoundingBox(
            x1Original.toFloat(),
            y1Original.toFloat(),
            x2Original.toFloat(),
            y2Original.toFloat(),
            detection.confidence,
            1
        )
        val center = Point(xCenterOriginal, yCenterOriginal)

        return Pair(boundingBox, center)
    }

    // Make YOLO bounding box thin
    fun drawBoundingBoxes(mat: Mat, box: BoundingBox) {
        val topLeft = Point(box.x1.toDouble(), box.y1.toDouble())
        val bottomRight = Point(box.x2.toDouble(), box.y2.toDouble())

        // Small thickness for YOLO bounding boxes
        val YOLO_BOX_THICKNESS = 2

        Imgproc.rectangle(
            mat,
            topLeft,
            bottomRight,
            Settings.BoundingBox.boxColor,
            YOLO_BOX_THICKNESS
        )

        val label = "User_1 (${("%.2f".format(box.confidence * 100))}%)"
        val fontScale = 0.6
        val textThickness = 1
        val baseline = IntArray(1)

        val textSize = Imgproc.getTextSize(
            label,
            Imgproc.FONT_HERSHEY_SIMPLEX,
            fontScale,
            textThickness,
            baseline
        )
        val textX = box.x1.toInt()
        val textY = (box.y1 - 5).toInt().coerceAtLeast(10)

        // Label background
        Imgproc.rectangle(
            mat,
            Point(textX.toDouble(), (textY + baseline[0]).toDouble()),
            Point((textX + textSize.width).toDouble(), (textY - textSize.height).toDouble()),
            Settings.BoundingBox.boxColor,
            Imgproc.FILLED
        )

        // Put text
        Imgproc.putText(
            mat,
            label,
            Point(textX.toDouble(), textY.toDouble()),
            Imgproc.FONT_HERSHEY_SIMPLEX,
            fontScale,
            Scalar(255.0, 255.0, 255.0),
            textThickness
        )
    }

    fun createLetterboxedBitmap(
        srcBitmap: Bitmap,
        targetWidth: Int,
        targetHeight: Int,
        padColor: Scalar = Scalar(0.0, 0.0, 0.0)
    ): Pair<Bitmap, Pair<Int, Int>> {
        val srcMat = Mat().also { Utils.bitmapToMat(srcBitmap, it) }
        val srcWidth = srcMat.cols().toDouble()
        val srcHeight = srcMat.rows().toDouble()
        val scale = min(targetWidth / srcWidth, targetHeight / srcHeight)

        val newWidth  = (srcWidth  * scale).toInt()
        val newHeight = (srcHeight * scale).toInt()

        val resized = Mat().also {
            Imgproc.resize(srcMat, it, Size(newWidth.toDouble(), newHeight.toDouble()))
            srcMat.release()
        }

        val padWidth  = targetWidth  - newWidth
        val padHeight = targetHeight - newHeight

        fun computePadding(total: Int): Pair<Int, Int> {
            val p1 = total / 2
            val p2 = total - p1
            return p1 to p2
        }

        val (top, bottom) = computePadding(padHeight)
        val (left, right) = computePadding(padWidth)

        val letterboxed = Mat().also {
            Core.copyMakeBorder(
                resized,
                it,
                top,
                bottom,
                left,
                right,
                Core.BORDER_CONSTANT,
                padColor
            )
            resized.release()
        }

        val outputBitmap = Bitmap.createBitmap(
            letterboxed.cols(),
            letterboxed.rows(),
            srcBitmap.config
        ).apply {
            Utils.matToBitmap(letterboxed, this)
            letterboxed.release()
        }

        return Pair(outputBitmap, Pair(left, top))
    }
}
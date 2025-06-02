@file:Suppress("SameParameterValue")

package com.developer27.xemotion.videoprocessing

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Rect
import android.util.Log
import com.developer27.xemotion.MainActivity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.apache.commons.math3.analysis.interpolation.SplineInterpolator
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction
import org.opencv.android.Utils
import org.opencv.core.*
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

// Define fontScale and thickness at class-level or inline
private const val fontScale = 2.0
private const val thickness = 3

// --------------------------------------------------
// Settings
// --------------------------------------------------
object Settings {

    object DetectionMode {
        enum class Mode {
            CONTOUR,
            YOLO
        }
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
        var splineStep = 0.01

        // Colors and thickness
        var originalLineColor = Scalar(173.0, 216.0, 230.0)    // Light Blue
        var splineLineColor   = Scalar(255.0, 203.0, 5.0) // Maize
        var lineThickness     = 100
    }

    object BoundingBox {
        var enableBoundingBox = true

        // Currently used for contour bounding boxes:
        var boxColor = Scalar(0.0, 39.0, 76.0)
        var boxThickness = 10
    }

    object Brightness {
        var factor = 2.0
        var threshold = 150.0
    }

    object ExportData {
        var frameIMG = false
        var enablePredictionLogging = false
    }

    object RollingShutter {
        // Set or update this from SettingsActivity
        var speedHz = 15f
    }
}

// --------------------------------------------------
// VideoProcessor
// --------------------------------------------------
class VideoProcessor(private val context: Context) {

    // We'll store the classification label here. If non-empty, we overlay it in the bounding box.
    var classificationLabel: String = ""

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

    // --------------------------------------------------------------------------------
    // ASYNCHRONOUS FRAME PROCESSING (for real-time camera usage, etc.)
    // --------------------------------------------------------------------------------
    fun processFrame(
        bitmap: Bitmap,
        callback: (Pair<Bitmap, Bitmap>?) -> Unit
    ) {
        // Example: Log the shutter speed each frame (just for demonstration):
        Log.d("VideoProcessor", "Current Rolling Shutter Speed = ${Settings.RollingShutter.speedHz} Hz")

        // We launch a coroutine on a background thread for asynchronous processing
        CoroutineScope(Dispatchers.Default).launch {
            val result: Pair<Bitmap, Bitmap>? = try {
                when (Settings.DetectionMode.current) {
                    Settings.DetectionMode.Mode.CONTOUR  -> processFrameInternalCONTOUR(bitmap)
                    Settings.DetectionMode.Mode.YOLO     -> processFrameInternalYOLO(bitmap)
                }
            } catch (e: Exception) {
                Log.e("VideoProcessor", "Error processing frame: ${e.message}", e)
                null
            }
            withContext(Dispatchers.Main) {
                callback(result)
            }
        }
    }

    // --------------------------------------------------------------------------------
    // 1) CONTOUR DETECTION
    // --------------------------------------------------------------------------------
    private fun processFrameInternalCONTOUR(bitmap: Bitmap): Pair<Bitmap, Bitmap>? {
        // 1) Preprocess the incoming bitmap to get a grayscale/filtered Mat (pMat).
        val (pMat, _) = Preprocessing.preprocessFrame(bitmap)

        // 2) Run contour detection: returns (center Point?, boundingRect Rect?, and the color frame Mat cMat).
        val (center, boundingRect, cMat) = ContourDetection.processContourDetection(pMat)

        // 3) If we have a valid bounding rectangle and a non‐empty label, draw the label + bounding box on cMat.
        if (boundingRect != null && classificationLabel.isNotEmpty()) {
            // 3a) Pick the textColor based on keywords
            val textColor = when {
                classificationLabel.contains("Sadness",    ignoreCase = true) -> Scalar(8.0,   223.0, 230.0) // Blue
                classificationLabel.contains("Anxious",    ignoreCase = true) -> Scalar(255.0, 255.0,   0.0) // Yellow
                classificationLabel.contains("Excitement", ignoreCase = true) -> Scalar(8.0,   230.0,  74.0) // Green
                classificationLabel.contains("Angry",      ignoreCase = true) -> Scalar(255.0, 102.0, 102.0) // Red
                else                                                           -> Scalar(255.0, 203.0,   5.0) // Maize
            }

            // 3b) Common margin to use if not in AR mode
            val margin = 20

            if (MainActivity.isArMode) {
                // -------- AR MODE: rotate the label and blit it above boundingRect --------

                // 1) Measure the text size (unrotated) exactly:
                val baselineArr = IntArray(1)
                val textSize: Size = Imgproc.getTextSize(
                    classificationLabel,                   // label text
                    Imgproc.FONT_HERSHEY_SIMPLEX,          // font face
                    fontScale,                             // font scale (Double)
                    thickness,                             // thickness (Int)
                    baselineArr                            // array to receive baseline
                )
                val baselinePx   = baselineArr[0]               // how far below baseline

                // 2) Compute overlay size by adding padding:
                val padding = 10
                val txtW = 600 + padding * 2
                val txtH = 600 + padding * 2 + baselinePx

                // 3) Create a tight 3‐channel BGR Mat (CvType.CV_8UC3) to render the text
                val textMat = Mat.zeros(txtH, txtW, CvType.CV_8UC3)

                // 4) Put the text INTO textMat at (padding, txtH - padding - baselinePx)
                Imgproc.putText(
                    textMat,
                    classificationLabel,
                    Point(padding.toDouble(), (txtH - padding - baselinePx).toDouble()),
                    Imgproc.FONT_HERSHEY_SIMPLEX,
                    fontScale,
                    textColor,
                    thickness
                )

                // 5) Rotate textMat 90° clockwise about its center:
                val centerPt = Point(txtW / 2.0, txtH / 2.0)
                val rotMat = Imgproc.getRotationMatrix2D(centerPt, 90.0, 1.0)
                val rotated = Mat()
                Imgproc.warpAffine(
                    textMat,
                    rotated,
                    rotMat,
                    Size(txtH.toDouble(), txtW.toDouble()),
                    Imgproc.INTER_LINEAR,
                    Core.BORDER_CONSTANT
                )

                // 6) Compute where to place “rotated” on cMat (centered horizontally, but directly below the box)
                val midX = (boundingRect.left + boundingRect.right) / 2
                val rawX = midX - rotated.cols() / 2
                val rawY = boundingRect.bottom - 350

                // 7) Clamp X
                val destX = rawX
                    .coerceAtLeast(0)
                    .coerceAtMost(cMat.width() - rotated.cols())

                // 8) Clamp Y
                val destY = rawY
                    .coerceAtLeast(0)
                    .coerceAtMost(cMat.height() - rotated.rows())

                // 9) Only overlay if rotated fits inside cMat
                if (rotated.cols() <= cMat.width() && rotated.rows() <= cMat.height()) {
                    val roiRect = org.opencv.core.Rect(destX, destY, rotated.cols(), rotated.rows())
                    val roiMat = cMat.submat(roiRect)
                    rotated.copyTo(roiMat)
                    roiMat.release()
                }

                // 10) Draw the bounding box on cMat (on top of everything):
                Imgproc.rectangle(
                    cMat,
                    Point(boundingRect.left.toDouble(),  boundingRect.top.toDouble()),
                    Point(boundingRect.right.toDouble(), boundingRect.bottom.toDouble()),
                    textColor,
                    Settings.BoundingBox.boxThickness
                )

                // 11) Release temps
                textMat.release()
                rotated.release()
            } else {
                // -------- NORMAL MODE: put non‐rotated text directly --------
                val baselineArrNorm = IntArray(1)
                val textSizeNorm: Size = Imgproc.getTextSize(
                    classificationLabel,
                    Imgproc.FONT_HERSHEY_SIMPLEX,
                    fontScale,
                    thickness,
                    baselineArrNorm
                )
                val textHeightNorm = textSizeNorm.height.toInt()

                val textX = boundingRect.left.toDouble()
                val textY = (boundingRect.top - margin - textHeightNorm.toDouble())
                    .coerceAtLeast(textHeightNorm.toDouble())

                Imgproc.putText(
                    cMat,
                    classificationLabel,
                    Point(textX, textY),
                    Imgproc.FONT_HERSHEY_SIMPLEX,
                    fontScale,
                    textColor,
                    thickness
                )

                Imgproc.rectangle(
                    cMat,
                    Point(boundingRect.left.toDouble(),  boundingRect.top.toDouble()),
                    Point(boundingRect.right.toDouble(), boundingRect.bottom.toDouble()),
                    textColor,
                    Settings.BoundingBox.boxThickness
                )
            }
        }

        // 4) Draw the center‐trace on cMat (if a center exists):
        TraceRenderer.drawTrace(center, cMat)

        // 5) Convert cMat → Bitmap so we can display the debug overlay:
        val debugOverlayBmp = Bitmap.createBitmap(cMat.cols(), cMat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(cMat, debugOverlayBmp)

        // 6) Also return the “cropped” region inside boundingRect (or full overlay if no box):
        val cropped: Bitmap = boundingRect
            ?.let { extractBoundingBoxRegion(bitmap, it) }
            ?: debugOverlayBmp

        // 7) Release Mats to avoid memory leaks:
        pMat.release()
        cMat.release()

        // 8) Return a Pair: (debugOverlay, croppedRegion)
        return debugOverlayBmp to cropped
    }

    // --------------------------------------------------------------------------------
    // 2) YOLO DETECTION (Asynchronous + synchronous helper)
    // --------------------------------------------------------------------------------
    private suspend fun processFrameInternalYOLO(bitmap: Bitmap): Pair<Bitmap, Bitmap> =
        withContext(Dispatchers.IO) {
            processFrameInternalYOLOSynchronous(bitmap)
        }

    /**
     * Synchronous YOLO for clarity. Returns (debugOverlay, letterboxed).
     */
    private fun processFrameInternalYOLOSynchronous(bitmap: Bitmap): Pair<Bitmap, Bitmap> {
        // 1) letterbox + model dims
        val (inputW, inputH, _) = getModelDimensions()
        val (letterboxed, offsets) = YOLOHelper.createLetterboxedBitmap(bitmap, inputW, inputH)

        // 2) convert original to Mat for drawing
        val m = Mat()
        Utils.bitmapToMat(bitmap, m)

        if (Settings.DetectionMode.enableYOLOinference && tfliteInterpreter != null) {
            // --- BEGIN FIXED OUTPUT SHAPE ALLOCATION ---
            val shape       = tfliteInterpreter!!.getOutputTensor(0).shape()
            val batch       = shape[0]        // should be 1
            val numBoxes    = shape[1]        // e.g. 3549
            val elemsPerBox = shape[2]        // e.g. 5 (x, y, w, h, conf)
            val out = Array(batch) { Array(numBoxes) { FloatArray(elemsPerBox) } }
            // --- END FIXED OUTPUT SHAPE ALLOCATION ---

            // prepare input and run inference
            val inputBuffer = TensorImage(DataType.FLOAT32).apply { load(letterboxed) }
            tfliteInterpreter!!.run(inputBuffer.buffer, out)

            // parse & draw your top detection
            YOLOHelper.parseTFLite(out)?.let { bestDetection ->
                val (box, centerPoint) = YOLOHelper.rescaleInferencedCoordinates(
                    bestDetection,
                    bitmap.width,
                    bitmap.height,
                    offsets,
                    inputW,
                    inputH
                )
                if (Settings.BoundingBox.enableBoundingBox) {
                    if (box.x2 > box.x1 && box.y2 > box.y1) {
                        YOLOHelper.drawBoundingBoxes(m, box, classificationLabel)
                    }
                }
                TraceRenderer.drawTrace(centerPoint, m)
            }
        }

        // convert Mat→Bitmap for overlay
        val debugOverlay = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(m, debugOverlay)
        m.release()

        return debugOverlay to letterboxed
    }

    // --------------------------------------------------------------------------------
    // Helper: get YOLO model dims
    // --------------------------------------------------------------------------------
    fun getModelDimensions(): Triple<Int, Int, List<Int>> {
        val inTensor = tfliteInterpreter?.getInputTensor(0)
        val inShape = inTensor?.shape()
        // Typically [1,416,416,3] => (batch=1, height=416, width=416, channels=3)
        val (height, width) = (
                inShape?.getOrNull(1) ?: 416
                ) to (
                inShape?.getOrNull(2) ?: 416
                )
        val outTensor = tfliteInterpreter?.getOutputTensor(0)
        val outShape = outTensor?.shape()?.toList() ?: listOf(1, 5, 3549)

        return Triple(width, height, outShape)
    }

    // --------------------------------------------------------------------------------
    // Helper: crop from original
    // --------------------------------------------------------------------------------
    private fun extractBoundingBoxRegion(srcBitmap: Bitmap, boundingRect: Rect): Bitmap {
        val left = boundingRect.left.coerceAtLeast(0)
        val top = boundingRect.top.coerceAtLeast(0)
        val right = boundingRect.right.coerceAtMost(srcBitmap.width)
        val bottom = boundingRect.bottom.coerceAtMost(srcBitmap.height)

        if (left >= right || top >= bottom) {
            // Invalid bounding box
            return Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888)
                .apply { eraseColor(Color.BLACK) }
        }
        val width = right - left
        val height = bottom - top
        return Bitmap.createBitmap(srcBitmap, left, top, width, height)
    }

    // --------------------------------------------------------------------------------
    // Export trace for data
    // --------------------------------------------------------------------------------
    fun exportTraceForDataCollection(): Bitmap {
        val snapshot = smoothDataList.toList()
        if (snapshot.isEmpty()) {
            // Return a tiny black bitmap if we have no data
            return Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888).apply {
                eraseColor(Color.BLACK)
            }
        }

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

        // Compute final integer dimensions
        val wDouble = width + 2.0 * padding
        val hDouble = height + 2.0 * padding
        val matWidth = max(1, wDouble.toInt())
        val matHeight = max(1, hDouble.toInt())

        val mat = Mat(matHeight, matWidth, CvType.CV_8UC4, Scalar(0.0, 0.0, 0.0, 255.0))

        val adjustedPoints = snapshot.map {
            Point((it.x - minX) + padding, (it.y - minY) + padding)
        }

        val origColor = Settings.Trace.splineLineColor
        val origThickness = Settings.Trace.lineThickness

        // Temporarily override color/thickness to get a white line on black
        Settings.Trace.splineLineColor = Scalar(255.0, 255.0, 255.0)
        Settings.Trace.lineThickness = 10

        TraceRenderer.drawSplineCurve(adjustedPoints, mat)

        // restore
        Settings.Trace.splineLineColor = origColor
        Settings.Trace.lineThickness = origThickness

        val intermediate = Bitmap.createBitmap(matWidth, matHeight, Bitmap.Config.ARGB_8888).apply {
            Utils.matToBitmap(mat, this)
            mat.release()
        }

        // final scale
        val finalWidth = 79
        val finalHeight = 68
        return Bitmap.createScaledBitmap(intermediate, finalWidth, finalHeight, true)
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
            Imgproc.threshold(eMat, it, Settings.Brightness.threshold, 255.0, Imgproc.THRESH_TOZERO)
            eMat.release()
        }
        val bMat = Mat().also {
            Imgproc.GaussianBlur(tMat, it, Size(5.0, 5.0), 0.0)
            tMat.release()
        }
        val k = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
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
    fun processContourDetection(mat: Mat): Triple<Point?, Rect?, Mat> {
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(mat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        hierarchy.release()

        if (contours.isEmpty()) {
            // Convert to BGR to keep consistent channel depth
            Imgproc.cvtColor(mat, mat, Imgproc.COLOR_GRAY2BGR)
            return Triple(null, null, mat)
        }

        val largestContour = contours.maxByOrNull { Imgproc.contourArea(it) }
            ?: run {
                Imgproc.cvtColor(mat, mat, Imgproc.COLOR_GRAY2BGR)
                return Triple(null, null, mat)
            }

        // Draw largest contour
        Imgproc.drawContours(
            mat,
            listOf(largestContour),
            -1,
            Settings.BoundingBox.boxColor,
            Settings.BoundingBox.boxThickness
        )

        val cvRect = Imgproc.boundingRect(largestContour)
        val boundingRect = Rect(
            cvRect.x,
            cvRect.y,
            cvRect.x + cvRect.width,
            cvRect.y + cvRect.height
        )

        if (Settings.BoundingBox.enableBoundingBox) {
            Imgproc.rectangle(mat, cvRect, Settings.BoundingBox.boxColor, Settings.BoundingBox.boxThickness)
        }

        val m = Imgproc.moments(largestContour)
        val center = Point(m.m10 / m.m00, m.m01 / m.m00)

        // Convert to BGR so we can overlay lines or text
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_GRAY2BGR)

        return Triple(center, boundingRect, mat)
    }
}

// --------------------------------------------------
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
            val conf    = rawOutput[0][4][i]

            if (conf >= Settings.Inference.confidenceThreshold) {
                detections.add(DetectionResult(xCenter, yCenter, width, height, conf))
            }
        }
        if (detections.isEmpty()) {
            return null
        }

        // Convert to bounding boxes
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

        return nmsDetections.maxByOrNull { it.confidence }
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

    /**
     * Rescale from letterbox-space back to original image coords.
     */
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

        val xCenterLetter = detection.xCenter * modelInputWidth
        val yCenterLetter = detection.yCenter * modelInputHeight
        val wLetter       = detection.width  * modelInputWidth
        val hLetter       = detection.height * modelInputHeight

        val xCenterOrig = (xCenterLetter - padLeft) / scale
        val yCenterOrig = (yCenterLetter - padTop)  / scale
        val wOrig       = wLetter / scale
        val hOrig       = hLetter / scale

        val x1 = xCenterOrig - (wOrig / 2)
        val y1 = yCenterOrig - (hOrig / 2)
        val x2 = xCenterOrig + (wOrig / 2)
        val y2 = yCenterOrig + (hOrig / 2)

        val boundingBox = BoundingBox(
            x1.toFloat(),
            y1.toFloat(),
            x2.toFloat(),
            y2.toFloat(),
            detection.confidence,
            1
        )
        val center = Point(xCenterOrig, yCenterOrig)
        return boundingBox to center
    }

    /**
     * Draw bounding box + classification label.
     * If MainActivity.isArMode == true, rotates all coords 90° CW.
     */
    fun drawBoundingBoxes(mat: Mat, box: BoundingBox, classificationLabel: String) {
        // Safety: skip degenerate
        if (box.x2 <= box.x1 || box.y2 <= box.y1) return

        // Always draw the rectangle normally
        val tl = Point(box.x1.toDouble(), box.y1.toDouble())
        val br = Point(box.x2.toDouble(), box.y2.toDouble())
        val textColor = when {
            classificationLabel.contains("Sadness",    ignoreCase = true) -> Scalar(8.0, 223.0, 230.0)
            classificationLabel.contains("Anxious",    ignoreCase = true) -> Scalar(255.0, 255.0,   0.0)
            classificationLabel.contains("Excitement", ignoreCase = true) -> Scalar(8.0, 230.0,  74.0)
            classificationLabel.contains("Angry",      ignoreCase = true) -> Scalar(255.0, 102.0, 102.0)
            else                                                         -> Scalar(255.0, 203.0,   5.0)
        }
        Imgproc.rectangle(mat, tl, br, textColor, 10)

        // Prepare label
        var label     = classificationLabel.ifEmpty { "Tracking" }
        val fontScale = 1.5
        val thickness = 2
        val baseline  = IntArray(1)
        val textSize  = Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX, fontScale, thickness, baseline)

        // Text position (normal)
        val margin = 50
        val textX  = tl.x.toInt()
        val textY  = (tl.y.toInt() - baseline[0] - margin)
            .coerceAtLeast((textSize.height + margin).toInt())

        if (MainActivity.isArMode) {
            // --- ROTATED TEXT IN AR MODE ---
            val extraMargin = 50
            val padding     = 30

            // Keep it square to simplify the rotation
            val minW = 600
            val minH = 600
            val txtW = max(minW + extraMargin, minH + extraMargin)
            val txtH = txtW

            // 1) Render label into a blank RGBA Mat
            val textMat = Mat.zeros(txtH, txtW, CvType.CV_8UC4)
            Imgproc.putText(
                textMat,
                label,
                Point(padding.toDouble(), txtH - padding.toDouble()),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                fontScale,
                textColor,
                thickness
            )

            // 2) Rotate that Mat 90° CW
            val center = Point(txtW / 2.0, txtH / 2.0)
            val rotMat = Imgproc.getRotationMatrix2D(center, 90.0, 1.0)
            val rotated = Mat()
            Imgproc.warpAffine(
                textMat,
                rotated,
                rotMat,
                Size(txtH.toDouble(), txtW.toDouble())
            )

            // 3) Compute where to place it (centered above the box)
            val midX   = ((box.x1 + box.x2) / 2.0).toInt()
            val floatY = (box.y1.toInt() - extraMargin).coerceAtLeast(rotated.rows() / 2)
            val destX  = (midX - rotated.cols() / 2).coerceAtLeast(0)
            val destY  = (floatY - rotated.rows() / 2).coerceAtLeast(0)

            // 4) Blit it onto the main mat if it fits
            if (destX + rotated.cols() <= mat.width() &&
                destY + rotated.rows() <= mat.height()
            ) {
                val roi = mat.submat(
                    org.opencv.core.Rect(destX, destY, rotated.cols(), rotated.rows())
                )
                rotated.copyTo(roi)
            }

            // Finally, bounding box on top
            val tl = Point(box.x1.toDouble(), box.y1.toDouble())
            val br = Point(box.x2.toDouble(), box.y2.toDouble())
            Imgproc.rectangle(mat, tl, br, textColor, 10)

        } else {
            // --- NORMAL TEXT AS BEFORE ---
            Imgproc.putText(
                mat,
                label,
                Point(textX.toDouble(), textY.toDouble()),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                fontScale,
                textColor,
                thickness
            )
        }
    }

    fun createLetterboxedBitmap(
        srcBitmap: Bitmap,
        targetWidth: Int,
        targetHeight: Int,
        padColor: Scalar = Scalar(0.0, 0.0, 0.0)
    ): Pair<Bitmap, Pair<Int, Int>> {
        val srcMat = Mat().also { Utils.bitmapToMat(srcBitmap, it) }
        val srcW   = srcMat.cols().toDouble()
        val srcH   = srcMat.rows().toDouble()
        val scale  = min(targetWidth / srcW, targetHeight / srcH)

        val newW = (srcW * scale).toInt()
        val newH = (srcH * scale).toInt()

        val resized = Mat().also {
            Imgproc.resize(srcMat, it, Size(newW.toDouble(), newH.toDouble()))
            srcMat.release()
        }

        val padW = targetWidth  - newW
        val padH = targetHeight - newH

        fun computePadding(total: Int): Pair<Int, Int> {
            val p1 = total / 2
            val p2 = total - p1
            return p1 to p2
        }

        val (top, bottom)  = computePadding(padH)
        val (left, right)  = computePadding(padW)
        val borderedMat    = Mat().also {
            Core.copyMakeBorder(resized, it, top, bottom, left, right, Core.BORDER_CONSTANT, padColor)
            resized.release()
        }

        val outputBitmap = Bitmap.createBitmap(borderedMat.cols(), borderedMat.rows(), srcBitmap.config)
            .apply {
                Utils.matToBitmap(borderedMat, this)
                borderedMat.release()
            }

        return outputBitmap to (left to top)
    }
}

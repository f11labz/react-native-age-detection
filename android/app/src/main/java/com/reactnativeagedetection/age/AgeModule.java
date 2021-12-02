package com.reactnativeagedetection.age;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Paint;
import android.os.Bundle;
import android.util.Base64;
import android.util.Log;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.TextView;

import com.facebook.react.bridge.Callback;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Vector;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

public class AgeModule extends ReactContextBaseJavaModule {
    private static ReactApplicationContext reactContext;
    private static final String TAG = "AGE MODULE";
    private CascadeClassifier cascadeClassifier;
    private Mat grayscaleImage;
    private int absoluteFaceSize;
    private int averageframes=5;
    private int frameCounter=0;
    List<MatOfRect> averageFaces = new ArrayList<MatOfRect>();

    static {
        System.loadLibrary("tensorflow_inference");
    }

    private static final String MODEL_FILE = "file:///android_asset/frozen_age_graph.pb";
    private static final String MODEL_LABELS = "file:///android_asset/age_labels.txt";

    private static final String INPUT_NODE = "input";
    private static final String OUTPUT_NODE = "output/output";

    private static final int IMAGE_MEAN = 117;
    private static final float IMAGE_STD = 1;

    private static final int MAX_FACES=6;

    //AGE_LIST = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
    private Vector<String> labels = new Vector<String>();
    //RESIZE_FINAL = 227
    private static final int IMAGE_INPUT_SIZE = 227;
    //Color Image? or grayscale and normalized?
    private static final int[] INPUT_SIZE = {IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 3};

    private TensorFlowInferenceInterface inferenceInterface;

    private static boolean GET_AGE = true;

    AgeModule(ReactApplicationContext context) {
        super(context);
        this.reactContext = context;
        OpenCVLoader.initDebug();
    }

    @Override
    public String getName() {
        return "VilaAgeModule";
    }


    @ReactMethod
    public void loadModel() {
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    load_cascade();
                    String actualFilename = MODEL_LABELS.split("file:///android_asset/")[1];
                    Log.i(TAG, "Reading labels from: " + actualFilename);
                    BufferedReader br = null;
                    try {
                        br = new BufferedReader(new InputStreamReader(reactContext.getAssets().open(actualFilename)));
                        String line;
                        while ((line = br.readLine()) != null) {
                            labels.add(line);
                        }
                        br.close();
                    } catch (IOException e) {
                        throw new RuntimeException("Problem reading label file!" , e);
                    }

                    Log.i("OnCreate", "Labels:" + labels.toArray().toString());
                    String modelActualFilename = MODEL_FILE.split("file:///android_asset/")[1];
                    Log.i("onCreate", "Reading model from: " + modelActualFilename);
                    inferenceInterface = new TensorFlowInferenceInterface(reactContext.getAssets(), MODEL_FILE);
                    // The shape of the output is [N, NUM_CLASSES], where N is the batch size.
                    final Operation operation = inferenceInterface.graphOperation(OUTPUT_NODE);
                    final int numClasses = (int) operation.output(0).shape().size(1);
                    Log.i(TAG, "Read " + labels.size() + " labels, output layer size is " + numClasses);
                } catch (final Exception e) {
                    //if they aren't found, throw an error!
                    throw new RuntimeException("Error initializing classifiers!", e);
                }
            }
        }).start();
    }


    private void load_cascade() {
        String actualFilename = "lbpcascade_frontalface.xml";
        // Copy the resource into a temp file so OpenCV can load it
        try (InputStream is = reactContext.getAssets().open(actualFilename)) {
            File cascadeDir = reactContext.getDir("cascades", Context.MODE_PRIVATE);
            File mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
            try (FileOutputStream os = new FileOutputStream(mCascadeFile)) {

                byte[] buffer = new byte[4096];
                int bytesRead;
                while ((bytesRead = is.read(buffer)) != -1) {
                    os.write(buffer, 0, bytesRead);
                }

                // Load the cascade classifier
                cascadeClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                if (!cascadeClassifier.load(mCascadeFile.getAbsolutePath())) {
                    Log.e("OpenCVActivity", "Failed to load cascade classifier");
                } else {
                    Log.i("OpenCVActivity", "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());
                }
            }
        } catch (Exception e) {
            Log.e("OpenCVActivity", "Error loading cascade", e);
            throw new RuntimeException("Error loading cascade resources", e);
        }
    }


    @ReactMethod
    public void detectAge(String path, final Callback callback){
        try {
            Log.i("detectAge", "starting detect age 1");
            byte[] decodedString = Base64.decode(path, Base64.DEFAULT);
            Bitmap image = BitmapFactory.decodeByteArray(decodedString, 0, decodedString.length);
            Bitmap grayImage = toGrayscale(image);
            // Create a grayscale image
            Mat colorImage = new Mat(image.getHeight(), image.getWidth(), CvType.CV_8UC4);
            Utils.bitmapToMat(image, colorImage);
            grayscaleImage = new Mat(image.getHeight(), image.getWidth(), CvType.CV_8UC4);
            //        //Mat greyImage = aInputFrame.gray();
            Imgproc.cvtColor(colorImage, grayscaleImage, Imgproc.COLOR_RGB2GRAY);
            //        Utils.bitmapToMat(grayImage, grayscaleImage);
            MatOfRect faces = new MatOfRect();
            // Use the classifier to detect faces
            absoluteFaceSize = (int) (image.getHeight() * .2);
            if (cascadeClassifier != null) {
                cascadeClassifier.detectMultiScale(grayscaleImage, faces, 1.1, 5, 2,
                        new Size(absoluteFaceSize, absoluteFaceSize), new Size());
            }

            // If there are any faces found, draw a rectangle around it
            Rect[] facesArray = faces.toArray();
            final List<Bitmap> faceBmps = new ArrayList<Bitmap>();
            //final List<Bitmap> greyFaceBmps = new ArrayList<Bitmap>();
            Log.i("detectAge", "starting detect age 2");
            Bitmap frameBmp = Bitmap.createBitmap(colorImage.width(), colorImage.height(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(colorImage, frameBmp);
            for (int i = 0; i < facesArray.length; i++) {
                Rect face = facesArray[i];
                Imgproc.rectangle(colorImage, face.tl(), face.br(), new Scalar(0, 255, 0, 255), 3);
                Bitmap faceBmp = Bitmap.createBitmap(frameBmp, (int) face.tl().x, (int) face.tl().y, face.width, face.height);
                faceBmps.add(faceBmp);
            }
            Log.i("detectAge", "starting detect age 3 " + Integer.toString(faceBmps.size()) + Boolean.toString(GET_AGE));
            //        if (GET_AGE) {
            //            GET_AGE = false;
            //            FaceProcessor ageProcessor = new FaceProcessor() {
            //                @Override
            //                public void processFace() {
            Log.i("detectAge", "starting detect age process 1 ");
            String[] outputNames = new String[]{OUTPUT_NODE};
            int[] intValues = new int[IMAGE_INPUT_SIZE * IMAGE_INPUT_SIZE];
            float[] floatValues = new float[IMAGE_INPUT_SIZE * IMAGE_INPUT_SIZE * 3];
            float[] outputs = new float[8];
            //final List<String> ages = new ArrayList<>();
            Log.i("detectAge", "starting detect age process 2 ");
            //Initailize to 0
            for (int idx = 0; idx < outputs.length; idx++) {
                outputs[idx] = 0;
            }

            Log.w("processFace", "Processing Faces!");
            String age = "";
            for (int idx = 0; idx < faceBmps.size(); idx++) {

                if (idx >= MAX_FACES) {
                    break;
                }

                Bitmap faceBmp = faceBmps.get(idx);
                Bitmap resizedFace = resize(faceBmp, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE);
                Log.i("processFace", "Normalizing Input Image:" + resizedFace.getWidth() + ", " + resizedFace.getHeight());
                resizedFace.getPixels(intValues, 0, resizedFace.getWidth(), 0, 0, resizedFace.getWidth(), resizedFace.getHeight());
                for (int i = 0; i < intValues.length; ++i) {
                    final int val = intValues[i];
                    floatValues[i * 3 + 0] = (((val >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD;
                    floatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD;
                    floatValues[i * 3 + 2] = ((val & 0xFF) - IMAGE_MEAN) / IMAGE_STD;
                }


                // Get age from TF model
                // Copy the input data into TensorFlow.
                Log.i("processFace", "Feeding Input to model");
                inferenceInterface.feed(INPUT_NODE, floatValues, 1, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 3);

                // Run the inference call.
                Log.i("processFace", "Running Model");
                inferenceInterface.run(outputNames, false);

                // Copy the output Tensor back into the output array.
                Log.i("processFace", "Fetching Output of Model");
                inferenceInterface.fetch(OUTPUT_NODE, outputs);

                Log.i("processFace", "Output Of Model:" + Arrays.toString(outputs));

                // Find Age
                int max_idx = 0;
                for (int idx_out = 0; idx_out < outputs.length; idx_out++) {
                    if (outputs[idx_out] > outputs[max_idx]) {
                        max_idx = idx_out;
                    }
                }

                age = labels.get(max_idx);

                Log.i("processFace", "Estimated Age(" + max_idx + "):" + labels.get(max_idx));
                //ages.add(labels.get(max_idx));
            }
            callback.invoke(null, age);
            //                }
            //            };
            //            try {
            //                faceProcessQueue.put(ageProcessor);
            //            } catch ( Exception ex ) {
            //                Log.e("OnCameraFrame", "Thread for face Processing has stopped!");
            //                callback.invoke("Thread for face Processing has stopped!");
            //            }
            //        } else {
            //            callback.invoke("Already age detection is processing");
            //        }
        } catch (Exception e) {
            callback.invoke(e.getMessage());
        }
    }

    private static Bitmap resize(Bitmap image, int maxWidth, int maxHeight) {
        if (maxHeight > 0 && maxWidth > 0) {
            int width = image.getWidth();
            int height = image.getHeight();
            float ratioBitmap = (float) width / (float) height;
            float ratioMax = (float) maxWidth / (float) maxHeight;

            int finalWidth = maxWidth;
            int finalHeight = maxHeight;
            if (ratioMax > ratioBitmap) {
                finalWidth = (int) ((float)maxHeight * ratioBitmap);
            } else {
                finalHeight = (int) ((float)maxWidth / ratioBitmap);
            }
            image = Bitmap.createScaledBitmap(image, finalWidth, finalHeight, true);
            return image;
        } else {
            return image;
        }
    }

    private interface FaceProcessor {
        void processFace();
    }

    class FaceConsumer implements Runnable {
        private final BlockingQueue<FaceProcessor> queue;
        FaceConsumer(BlockingQueue q) { queue = q; }
        public void run() {
            try {
                while (true) { consume(queue.take()); }
            } catch (InterruptedException ex) {
                Log.i("FaceConsumer", "Face Processing has stopped");
                return;
            }
        }

        void consume(FaceProcessor aFaceProcessor) {
            aFaceProcessor.processFace();
        }
    }
    public Bitmap toGrayscale(Bitmap bmpOriginal)
    {
        int width, height;
        height = bmpOriginal.getHeight();
        width = bmpOriginal.getWidth();

        Bitmap bmpGrayscale = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        Canvas c = new Canvas(bmpGrayscale);
        Paint paint = new Paint();
        ColorMatrix cm = new ColorMatrix();
        cm.setSaturation(0);
        ColorMatrixColorFilter f = new ColorMatrixColorFilter(cm);
        paint.setColorFilter(f);
        c.drawBitmap(bmpOriginal, 0, 0, paint);
        return bmpGrayscale;
    }
}

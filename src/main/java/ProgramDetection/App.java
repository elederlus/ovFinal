package ProgramDetection;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * @author Adam Gibson
 */
public class App {
    private static final Logger log = LoggerFactory.getLogger(App.class);
    public static void main(String[] args) throws IOException, InterruptedException
    {
        final int numRows = 256;
        final int numColumns = 256;
        int nChannels = 1;
        int outputNum = 2;
        int batchSize = 10;
        int iterations = 10;
        int splitTrainNum = (int) (batchSize*.8);
        int seed = 123;
        int listenerFreq = iterations/2;

        
        DataSet ovNext;
        SplitTestAndTrain trainTest;
        DataSet trainInput;
        List<INDArray> testInput = new ArrayList<>();
        List<INDArray> testLabels =  new ArrayList<>();
        // Set path to the labeled images
        String labeledPath = System.getProperty("user.home")+"/Desktop/ov";
        //create array of strings called labels
        List<String> labels = new ArrayList<>();
        //traverse dataset to get each label
        for(File f : new File(labeledPath).listFiles()) {
            labels.add(f.getName());
            System.out.println(f.getName());
        }
        log.info("Load data....");
        // DataSetIterator lfw = new LFWDataSetIterator(batchSize, numSamples);
        // Instantiating RecordReader. Specify height and width of images.
        RecordReader recordReader = new ImageRecordReader(256, 256, true, labels);
        // Canova to DL4J
        DataSetIterator ov = new RecordReaderDataSetIterator(recordReader, 65536, labels.size());
        // Point to data path.

        try {
            recordReader.initialize(new FileSplit(new File(labeledPath)));
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        log.info("Build model....");
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .regularization(true).l2(0.0005)
                .learningRate(0.01)//.biasLearningRate(0.02)
                //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list(6)
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation("identity")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(50)
                        .activation("identity")
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation("relu")
                        .nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false);


        new ConvolutionLayerSetup(builder,numRows,numColumns,nChannels);
        MultiLayerNetwork model = new MultiLayerNetwork(builder.build());
        model.init();

        log.info("Train model....");
        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));
        while(ov.hasNext()) {
            ovNext = ov.next();
            ovNext.scale();
            trainTest = ovNext.splitTestAndTrain(splitTrainNum, new Random(seed)); // train set that is the result
            trainInput = trainTest.getTrain(); // get feature matrix and labels for training
            testInput.add(trainTest.getTest().getFeatureMatrix());
            testLabels.add(trainTest.getTest().getLabels());
            model.fit(trainInput);
        }
        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum);
        for(int i = 0; i < testInput.size(); i++) {
            INDArray output = model.output(testInput.get(i));
            eval.eval(testLabels.get(i), output);
        }
        INDArray output = model.output(testInput.get(0));
        eval.eval(testLabels.get(0), output);
        log.info(eval.stats());
        log.info("**************** finished ********************");
    }
}
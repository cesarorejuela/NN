import java.util.Arrays;

public class NN_Structure {
    Weights_RandomStarter weights = new Weights_RandomStarter();
    Feed_Forward calculate_feed = new Feed_Forward();
    BackPropagation calculate_backPropagation = new BackPropagation();

    int inputLayer_Size;
    int hiddenLayer_Size;
    int outputLayer_Size;
    double[][] set_Inputs;
    double[][] weights_Set1_In_Hidden;
    double[][] weights_Set2_Hidden_Output;
    double learningRate;

    public NN_Structure(int inputLayer_Size, int hiddenLayer_Size, int outputLayer_Size, double[][] set_Inputs, double learningRate) {
        this.inputLayer_Size = inputLayer_Size;
        this.hiddenLayer_Size = hiddenLayer_Size;
        this.outputLayer_Size = outputLayer_Size;
        this.set_Inputs = set_Inputs;
        this.learningRate = learningRate;
        this.weights_Set1_In_Hidden = new double[inputLayer_Size][hiddenLayer_Size];
        this.weights_Set2_Hidden_Output = new double[hiddenLayer_Size][outputLayer_Size];
        start_Random_Weights();
    }

    private void start_Random_Weights() {
        System.out.println("Initializing Network with desired parameters and random weights...");
        weights.weights_Generator(weights_Set1_In_Hidden);
        weights.weights_Generator(weights_Set2_Hidden_Output);
    }

    // Method only for testing with updated weights_Set1_In_Hidden & weights_Set2_Hidden_Output
    public double[] feed_FORWARD_For_Testing(double[] given_Inputs, double threshold) {
        double[][] hiddenOutputs = calculate_feed.feedForwardLayer(given_Inputs, weights_Set1_In_Hidden);
        double[][] finalOutputs = calculate_feed.feedForwardLayer(hiddenOutputs[1], weights_Set2_Hidden_Output);

        //we can also apply a threshold value from Main method to classify output as 1 or 0
        double[] classifiedOutput = new double[finalOutputs[1].length];
        for (int i = 0; i < finalOutputs[1].length; i++) {
            classifiedOutput[i] = (finalOutputs[1][i] >= threshold) ? 1.0 : 0.0;
        }
        // Return the classified output value (success or failure)
        return classifiedOutput;
    }

    public void train(double[][] inputs, double[] targets, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            System.out.println("Epoch # " + (epoch + 1));

            for (int i = 0; i < inputs.length; i++) {
                System.out.println();
                System.out.println("Input # " + (i + 1));

                // Using the feedForwardForTraining method only for training and doing first feedforward with random weights
                double[][] outputs = calculate_feed.feedForwardForTraining(inputs[i], weights_Set1_In_Hidden, weights_Set2_Hidden_Output);

                //Storing values in arrays of weightedSums and their corresponding activations levels to be used as in formulas
                double[] hiddenWeightedSums = outputs[0];
                double[] finalWeightedSums = outputs[1];
                double[] hiddenOutputs = outputs[2];
                double[] finalOutputs = outputs[3];

                // Print weights and outputs for a log file
                System.out.println("Updating weights_Set1_Input_Hidden_Layers continuously");
                System.out.println(Arrays.deepToString(weights_Set1_In_Hidden));
                System.out.println("Updating weights_Set2_Hidden_Output_Layers continuously");
                System.out.println(Arrays.deepToString(weights_Set2_Hidden_Output));

                // Backpropagation phase to be called in Main class as part of train method of NN_Structure
                calculate_backPropagation.formula_backpropagation(
                        inputs[i], hiddenWeightedSums, finalWeightedSums, finalOutputs[0], targets[i],
                        weights_Set1_In_Hidden, weights_Set2_Hidden_Output, learningRate, hiddenOutputs
                );
            }
        }
    }
}

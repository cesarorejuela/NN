public class Feed_Forward {

    //Method to received weighted sums of all network's units to apply this formula for obtaining activation values
    private double sigmoid_Function(double x) {
        return 1 / (1 + Math.exp(-x));
    }


    // main feed_forward method with formula, to be used for testing and for complementing feedForwardForTraining method
    public double[][] feedForwardLayer(double[] inputs, double[][] weights) {
        double[] activations = new double[weights[0].length];
        double[] weightedSums = new double[weights[0].length];

        for (int j = 0; j < weights[0].length; j++) {
            double sum = 0;
            for (int i = 0; i < inputs.length; i++) {

                //Formula feed_Forward
                sum += inputs[i] * weights[i][j];
            }
            weightedSums[j] = sum;
            //calling and applying sigmoid_Function method
            activations[j] = sigmoid_Function(sum);
        }
        return new double[][] {weightedSums, activations};
    }

    // Feedforward method only for training which returns both hidden and final outputs with or without activation level
    public double[][] feedForwardForTraining(double[] inputs, double[][] weights1, double[][] weights2) {
        double[][] hiddenLayerOutput = feedForwardLayer(inputs, weights1);
        double[][] finalOutput = feedForwardLayer(hiddenLayerOutput[1], weights2);

        // Return 4 values; hidden and final outputs of weighted sums, and hidden and final outputs applying sigmoid
        return new double[][] {hiddenLayerOutput[0], finalOutput[0], hiddenLayerOutput[1], finalOutput[1]};
    }
}

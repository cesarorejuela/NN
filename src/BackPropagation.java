import java.util.Arrays;

public class BackPropagation {

    // Derivative of the sigmoid function, used in backpropagation formulas
    // g′(x)=σ(x)×(1−σ(x))
    private double sigmoid_Derivative(double weightedSum) {
        double sigmoid = 1 / (1 + Math.exp(-weightedSum)); // Calculate sigmoid from weighted sum
        return sigmoid * (1 - sigmoid); // Derivative of sigmoid
    }

    // Backpropagation to calculate the error gradients for both layers (output-hidden, hidden-input)
    public void formula_backpropagation(
            double[] inputs, double[] hiddenWeightedSums, double[] finalWeightedSums, double finalOutput, double target,
            double[][] weights1, double[][] weights2, double learningRate, double[] hiddenOutputs) {

        // Calculate the error (delta) at each output node
        // Err = t - o
        double outputError = target - finalOutput;

        // Δerror(outputDelta) = Err * g′(finalWeightedSums)
        double outputDelta = outputError * sigmoid_Derivative(finalWeightedSums[0]); // Use pre-activation value g′(in-final)
        //g′(in) is the derivative of the activation function evaluated with the weighted sum
        System.out.println("Output Error: " + outputError);
        System.out.println("Output Delta: " + outputDelta);

        // Calculate the hidden layer error (backpropagate the error) at each hidden unit
        double[] hiddenDeltas = new double[hiddenWeightedSums.length];
        for (int j = 0; j < hiddenWeightedSums.length; j++) {

            // Δerror(hiddenDeltas) = g′(hiddenWeightedSums) * w2 * Δerror(outputDelta)
            double hiddenError = outputDelta * weights2[j][0];
            hiddenDeltas[j] = hiddenError * sigmoid_Derivative(hiddenWeightedSums[j]); // Use pre-activation g′(in-hidden)
        }
        System.out.println("Hidden Deltas: " + Arrays.toString(hiddenDeltas));

        // Formula using Δerror(outputDelta) to update weights2 between hidden and output layer
        // w2 = w2initial + ( alpha * Δerror(outputDelta) *  hiddenOutputs )
        for (int j = 0; j < weights2.length; j++) {
            weights2[j][0] += learningRate * outputDelta * hiddenOutputs[j]; // Use activation for weight update
        }
        System.out.println("Updated weights from hidden to output layer: " + Arrays.deepToString(weights2));

        // Formula using Δerror(hiddenDeltas)  to update weights1 between input and hidden layer
        // w1 = w1initial + ( alpha * Δerror(hiddenDeltas) * inputs )
        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < hiddenWeightedSums.length; j++) {
                weights1[i][j] += learningRate * hiddenDeltas[j] * inputs[i]; // Use activation for weight update
            }
        }
        System.out.println("Updated weights from input to hidden layer: " + Arrays.deepToString(weights1));
    }
}

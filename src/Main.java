import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        double[][] set_Inputs = {
                {0.0, 0.0},
                {0.0, 1.0},
                {1.0, 0.0},
                {1.0, 1.0}
        };

        double[] targets = {0.0, 1.0, 1.0, 1.0}; // OR target outputs

        //Initializing the structure of the neural network using parameters defined in NN_Structure
        NN_Structure logic_OperatorNN = new NN_Structure(2, 4, 1, set_Inputs, 0.5);

        // First,Training the network using train method from NN_Structure (n epochs until getting the desire output)
        logic_OperatorNN.train(set_Inputs, targets, 1);


        // Second, Testing the network with updated weights after using the train method (backpropagation) from NN_Structure
        double threshold = 0.5;
        System.out.println("\nOutputs after training:");
        for (int i = 0; i < set_Inputs.length; i++) {
            System.out.println("Input: " + Arrays.toString(set_Inputs[i]));
            System.out.println("Classified Output: " + Arrays.toString(logic_OperatorNN.feed_FORWARD_For_Testing(set_Inputs[i], threshold)));
        }
    }
}

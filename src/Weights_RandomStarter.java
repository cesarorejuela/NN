import java.util.Random;

public class Weights_RandomStarter {
    private Random randomly = new Random();

    public void weights_Generator(double[][] weight_Sets_any) {
        for (int i = 0; i < weight_Sets_any.length; i++) {
            for (int j = 0; j < weight_Sets_any[0].length; j++) {
                weight_Sets_any[i][j] = randomly.nextGaussian();
            }
        }
    }
}

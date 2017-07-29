package examples.mnist;
import java.text.DecimalFormat;

import nnetwork.NNetworkUtils;
import nnetwork.Network;

public class NeuralMNIST {

	static int inputSize  = 28 * 28;
	static int[] hiddenSize = {28, 28, 28};
	static int outputSize = 10;
	static double learningRate = 0.1;
	static DecimalFormat df = new DecimalFormat("##.##%");

	public static void main(String[] args) throws InterruptedException {
		MNISTParser train = new MNISTParser("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
		MNISTParser test = new MNISTParser("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");

		Network network = new Network(inputSize, hiddenSize, outputSize);


		for(int runs = 1; ; runs++) {

			/** TRAINING AREA */
			for(int i = 0; i < train.size(); i++) {
				boolean[][] image = train.getBinaryImage(i);
				double[] input = toInput(image);
				network.compute(input);
				
				int label = train.getLabel(i);
				double[] expected = NNetworkUtils.oneHotArray(outputSize, label);
				network.backpropagate(expected, learningRate);
				// if(i % 10000 == 0) System.out.println("Completed " + (i * 10000) + " iterations");
			}

			/** TESTING AREA */
			double confidence = 0, guessRate = 0;
			for(int i = 0; i < test.size(); i++) {
				double[] input = toInput(test.getBinaryImage(i));
				double[] output = network.compute(input);
				int guess = NNetworkUtils.softmax(output);
				if (guess == test.getLabel(i)) {
					confidence += output[guess] ;
					guessRate++;
				}
			}

			System.err.println("Iteration: " + runs);
			System.err.flush();
			Thread.sleep(100);
			System.out.println("\t Confidence:\t" + df.format(confidence / test.size())   + " of " + test.size());
			System.out.println("\t Guess Rate:\t" + df.format(guessRate / test.size())  + " of " + test.size());
			System.out.flush();	
		}
	}



	private static double[] toInput(boolean[][] image) {
		double[] input = new double[inputSize];
		for(int i = 0; i < 28; i++)
			for(int j = 0; j < 28; j++) 
				input[i + 28 * j] = image[i][j] ? 1 : 0;

		return input;
	}

	private static double sigmoid(double x) {
		return Math.tanh(x);
	}

}

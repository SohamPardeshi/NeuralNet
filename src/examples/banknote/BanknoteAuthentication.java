package examples.banknote;
import java.io.*;
import java.nio.file.*;
import java.util.Arrays;
import java.util.Collections;

import nnetwork.*;

/***
 * @author soham
 * Dataset courtesy of UCI Machine Learning Repository
 * Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
 *
 */

public class BanknoteAuthentication {
	public static void main(String[] Args) throws IOException{
		double[][] data = processFile("dataset/banknote");
		Collections.shuffle(Arrays.asList(data)); // Randomize data order

		Network net = new Network(4, new int[]{3}, 2);

		int iter = 0;
		int lim = 10;
		while(true){

			for(int i = 0; i < lim; i++){
				double[] input = Arrays.copyOf(data[i], 4);
				int label = (int)data[i][4];

				// Creates a 
				double[] expected = NNetworkUtils.oneHotArray(2, label);
				double learningRate = 0.1;

				net.compute(input);
				net.backpropagate(expected, learningRate);
			}

			int correct = 0;
			for(int i = lim; i < data.length; i++){
				double[] input = Arrays.copyOf(data[i], 4);
				int label = (int)data[i][4];

				double[] output = net.compute(input);

				int guess = NNetworkUtils.softmax(output);
				if(guess == label) correct++;
			}
			
			System.out.println("Iteration " + iter++ + ": ");
			System.out.println("\tClassified " + correct + " out of " + (data.length - lim));
		}

	}

	private static double[][] processFile(String file) throws IOException {
		// Convert file to double[][] 
		String f = new String(Files.readAllBytes(Paths.get(file)));
		String[] lines = f.split("\n");
		String[][] data = new String[lines.length][];

		double[][] ddata = new double[lines.length][];

		int count = 0;
		for(String line : lines){
			String[] vals = line.split(",");
			double[] dval = new double[vals.length];

			for(int i = 0; i < vals.length; i++)
				dval[i] = Double.parseDouble(vals[i]);

			ddata[count++] = dval;
		}
		
		return ddata;
	}
}

package nnetwork;

public class NNetworkUtils {
	public static int softmax(double[] output) {
		double max = -1; int pos = 0;
		for(int i = 0; i < output.length; i++)
			if(output[i] > max)
				max = output[(pos = i)];

		return pos;
	}
	public static double[] oneHotArray(int length, int hotIndex) {
		double[] array = new double[length];
		array[hotIndex] = 1;
		return array;
	}

}

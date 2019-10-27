package model;

public class Perplexity {
	public static double lda_training_perplexity(int[][] docword, double[][] theta,
			double[][] phi) {

		double perplexity = 0;

		int denominator = 0;

		for (int d = 0; d < docword.length; d++) {

			denominator += (docword[d].length);

		}

		double numerator = 0;

		for (int d = 0; d < docword.length; d++) {

			for (int n = 0; n < docword[d].length; n++) {

				double prob = 0;

				for (int k = 0; k < phi.length; k++) {

					prob += theta[d][k] * phi[k][docword[d][n]];

				}

				numerator += Math.log(prob);

			}

		}
		perplexity = Math.exp(-numerator / denominator);

		return perplexity;

	}
}

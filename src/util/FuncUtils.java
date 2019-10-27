package util;

import java.util.Random;

import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.CholeskyDecomposition;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularMatrixException;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.Well19937c;
import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.util.MathArrays;

import Jama.Matrix;



public class FuncUtils {
	/**
	 * Sample a value from a double array
	 * 
	 * @param probs 
	 * @return
	 */
	public static int rouletteGambling(double[] prob){
		int topic = 0;
		for (int k = 1; k < prob.length; k++) {
			prob[k] += prob[k - 1];
		}
		double u = Math.random() * prob[prob.length - 1];
		for (int t = 0; t < prob.length; t++) {
			if (u < prob[t]) {
				topic = t;
				break;
			}
		}
		return topic;
	}
	/**
	 * Sample a value from a double array
	 * 
	 * @param probs 
	 * @return topic number
	 */
	public static int rouletteGambling(double[][] prob){
		int K = prob.length;
		int A = prob[0].length;
		double[] pr_sum = new double[K * A];
		for (int k = 0; k < K; k++) {
			for (int a = 0; a < A; a++) {
				pr_sum[k  + a*K] = prob[k][a];
			}
		}
		int idx = rouletteGambling(pr_sum);
		return idx;
	}
	/**
	 * transpose of two-dimensional array
	 * 
	 * @param prob 
	 * @return
	 */
	public static double[][] arrayTrans(double[][] prob){
		double[][] pro_new =  new double[prob[0].length][prob.length];
		for(int i = 0; i < prob[0].length; i++){
			for(int j = 0; j < prob.length; j++){
				pro_new[i][j] = prob[j][i];
			}
		}
		return pro_new;
	}
	/**
	 * get the index of max value in an array
	 * 
	 * @param array
	 * @return the index of max value
	 */
	public static int maxValueIndex(double[] array) {
		double max = array[0];
		int maxVIndex = 0;
		for (int i = 1; i < array.length; i++) {
			if (array[i] > max) {
				max = array[i];
				maxVIndex = i;
			}
		}
		return maxVIndex;
	}

	public static double[] getGaussianSample(int K, double mean, double deviation) {
		Random r = new Random();
		double[] sample = new double[K];
		for(int k = 0; k < K; k ++) {
			sample[k] = r.nextGaussian() * Math.sqrt(deviation) + mean;
		}
		return sample;
	}
	/**sample from multivariate normal distribution
	 * @param mean
	 * @param identity array
	 * ****/
	public static double [] sampleFromMultivariateDistribution(double [] mean, double [][] variance){
		MultivariateNormalDistribution cc = new MultivariateNormalDistribution(mean, variance);
		double[] sampleValues = cc.sample();
		return sampleValues;
	}
	/**generate identity array
	 * [[100.   0.   0.   0.   0.]
	 * [  0. 100.   0.   0.   0.]
	 * [  0.   0. 100.   0.   0.]
	 * [  0.   0.   0. 100.   0.]
	 * [  0.   0.   0.   0. 100.]]
	 * @param identity 100
	 * @param dimension 5
	 * ****/
	public static double [][] generateIdentityArray(double identity,int dimension){
		double arr[][] = new double[dimension][dimension];
		for(int i = 0; i < arr.length; i++) {
			arr[i][i] = identity;  
		}
		return arr;
	}
	/**generate mean of multivariate normal distribution
	 * [0. 0. 0. 0. 0.]
	 * @param mean 0 
	 * @param dimension 4
	 * ****/
	public static double [] generateMeanArray(double mean, int dimension){
		double arr[] = new double[dimension];
		for (int i = 0; i < arr.length; i++) {
			arr[i] = mean;
		}
		return arr;
	}
	/**generate random array
	 * 
	 * @param row
	 * @param column
	 * @param value
	 * if value=1.0, we can get:
	 * [[0.024027   0.168236   0.25728113  0.55045508]
	 * [ 0.263268324  0.28504307   0.290910533  0.16077806]
	 * [ 0.0800969   0.3252087   0.28674675   0.307947605]
	 * ****/
	public static double [][] generateRandomSumOneArray(int row, int column,double value){
		RandomGenerator rg = new JDKRandomGenerator();
		double array[][] = new double[row][column];
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < column; j++) {
				array[i][j] = rg.nextDouble();
			}
			array[i] = MathArrays.normalizeArray(array[i],value);
		}
		return array;
	}
	/**get the inverse matrix of the input Matrix  A
	 * 
	 * @param RealMatrix A
	 * 
	 * ****/
	public static RealMatrix inverseMatrix(RealMatrix A) {
		RealMatrix result = new LUDecomposition(A).getSolver().getInverse();
		return result; 
	}
	/**The probability density function of the Multivariate t distributions
	 * Reference: Conjugate Bayesian analysis of the Gaussian distribution
	 * 
	 * 
	 * @param ArrayRealVector dataPoint
	 * @param ArrayRealVector meansVector
	 * @param RealMatrix covarianceMatrix
	 * @param double degreesOfFreedom
	 * @return The probability value
	 * 
	 * 
	 * @author Qianyang 
	 * ****/
	/*public static double multivariateTDensity(ArrayRealVector dataPoint, ArrayRealVector meansVector, RealMatrix covarianceMatrix, double degreesOfFreedom){
		LUDecomposition covariance = new LUDecomposition(covarianceMatrix);
//		System.out.println(covarianceMatrix);
		double logprob_left = Gamma.logGamma((degreesOfFreedom + dataPoint.getDimension())/2.0) - 
				(Gamma.logGamma(degreesOfFreedom / 2.0) + 0.5 * Math.log(covariance.getDeterminant()) + 
						dataPoint.getDimension()/2.0 * (Math.log(degreesOfFreedom) + Math.log(Math.PI)));		
		// compute x-u
		ArrayRealVector var = dataPoint.add(meansVector.mapMultiplyToSelf(-1.0));
//		System.out.println(var);
		// (x-u) to  matrix
		RealMatrix realMatrix = new Array2DRowRealMatrix(var.getDataRef());
		//compute left
		double logprob_right = (degreesOfFreedom + dataPoint.getDimension())/2.0 * Math.log(1 + realMatrix.transpose().multiply(new LUDecomposition(covarianceMatrix).getSolver().getInverse())
				.multiply(realMatrix).getData()[0][0]/degreesOfFreedom);
		System.out.println("left:" + logprob_left + "\tright:" + logprob_right);
		System.out.println(Math.exp(logprob_left -logprob_right));
		return Math.exp(logprob_left - logprob_right);
	}*/
	public static double multivariateTDensity(ArrayRealVector dataPoint, ArrayRealVector meansVector, RealMatrix covarianceMatrix, double degreesOfFreedom){
		LUDecomposition covariance = new LUDecomposition(covarianceMatrix);
		double logprob_left = Gamma.logGamma((degreesOfFreedom + dataPoint.getDimension())/2.0) - 
				(Gamma.logGamma(degreesOfFreedom / 2.0) + 0.5 * Math.log(covariance.getDeterminant()) + 
						dataPoint.getDimension()/2.0 * (Math.log(degreesOfFreedom) + Math.log(Math.PI)));		
		// compute x-u
		ArrayRealVector var = dataPoint.add(meansVector.mapMultiplyToSelf(-1.0));
		// (x-u) to  matrix
		RealMatrix realMatrix = new Array2DRowRealMatrix(var.getDataRef());
		//compute left
		double logprob_right = Math.log(1 + realMatrix.transpose().multiply(new LUDecomposition(covarianceMatrix).getSolver().getInverse())
				.multiply(realMatrix).getData()[0][0]/degreesOfFreedom);
//		System.out.println(logprob_left -(degreesOfFreedom + dataPoint.getDimension())/2.0 * logprob_right);
		System.out.println(Math.exp(logprob_left -(degreesOfFreedom + dataPoint.getDimension())/2.0 * logprob_right));
		return Math.exp(logprob_left -(degreesOfFreedom + dataPoint.getDimension())/2.0 * logprob_right);
	}
	/**
	 * 
	 * 多元t分布的概率密度函数
	 * 
	 * @param vector
	 * @param freedom
	 * @param mean
	 * @param sigma
	 * @return
	 */
	public static double multivariateTdistribution(double[] vector, double[] mean, double[][] sigma, double freedom) {

		double dimension = vector.length;

		Matrix s = new Matrix(sigma);

		Matrix m = new Matrix(mean, 1);

		Matrix v = new Matrix(vector, 1);

		Matrix c = v.minus(m);

		Matrix inverse = s.inverse();

		Matrix temp = c.times(inverse).times(c.transpose());

		double det = s.det();
		
		double prob = Gamma.gamma((freedom + dimension) / 2)
				/ (Gamma.gamma(freedom / 2) * Math.pow(freedom, dimension / 2) * Math.pow(Math.PI, dimension / 2))
				* Math.pow(det, -0.5) * Math.pow(1 + temp.det() / freedom, -(dimension + freedom) / 2);
		System.out.println("pro:" + prob);
		if (Double.isNaN(prob)) {
			
		}
		return prob;

	}
	/**Arrays  Search
	 * 
	 * @param arr
	 * @param targetValue
	 * @return boolean
	 * For example: arr = new int[] { 3, 5, 7, 11, 13 }
	 * targetValue = 3
	 * the return will be true
	 * ****/
	public static boolean arrSearch(int[] arr, int targetValue) {
		for (Integer s : arr) {
			if (s.equals(targetValue))
				return true;
		}
		return false;
	}
	/**Returns the mean vector of the input matrix by row or column
	 * 
	 * @param matrix  the input matrix
	 * @param rowOrColumn  if rowOrColumn = true, it represents row
	 * @return ArrayRealVector  the mean vector
	 * 
	 * ****/
	public static ArrayRealVector meanMatrix(RealMatrix matrix, boolean rowOrColumn){
		double mean[] ;
		if (rowOrColumn) {  // for row
			mean = new double[matrix.getRowDimension()];
			for (int i = 0; i < matrix.getRowDimension(); i++) {
				mean[i] = StatUtils.mean(matrix.getRowVector(i).toArray());
			}
		}else {  // for column
			mean = new double[matrix.getColumnDimension()];
			for (int i = 0; i < matrix.getColumnDimension(); i++) {
				mean[i] = StatUtils.mean(matrix.getColumnVector(i).toArray());
			}
		}
		return (new ArrayRealVector(mean));
		
	}
	/**sample from wishart distribution
	 * @param   df  
	 * @param   scaleMatrix 
	 * @return  covariance matrix
	 * 
	 * blog:http://blog.csdn.net/wjj5881005/article/details/53535613
	 * ****/
	public static RealMatrix sampleFromInverseWishartDistribution(double df, RealMatrix scaleMatrix){
		//get the inverse matrix of scale matrix
		RealMatrix inverseMatrix = inverseMatrix(scaleMatrix);  
		RealMatrix symmetricInverseMatrix = convertMatrixToSymmetry(inverseMatrix);
		// sampling
		for (int i = 0; i < 1000; i++) {
			try {
				RealMatrix samples = sampleFromWishartDistribution(df, symmetricInverseMatrix);
				RealMatrix inverse_samples = inverseMatrix(samples);
				RealMatrix symmetric_inverse_samples = convertMatrixToSymmetry(inverse_samples);
				return symmetric_inverse_samples;
			} catch (SingularMatrixException ex) {
				ex.printStackTrace();
			}
		}
		throw new RuntimeException("Unable to generate inverse wishart samples!");
	}
	/**sample from wishart distribution
	 * @param   df  
	 * @param   scaleMatrix 
	 * @return  p dimensional symmetric positive definite matrix
	 * 
	 * blog:http://blog.csdn.net/wjj5881005/article/details/53535613
	 * ****/
	public static RealMatrix sampleFromWishartDistribution(double df, RealMatrix scaleMatrix){
		RandomGenerator random = new Well19937c();
		int dim = scaleMatrix.getColumnDimension(); // get the dimension of the matrix
		GammaDistribution [] gammas = new GammaDistribution[dim];
		for (int i = 0; i < dim; i++) {
			gammas[i] = new GammaDistribution((df-i+0.0)/2, 2);
		}
		CholeskyDecomposition cholesky = new CholeskyDecomposition(scaleMatrix);
		// Build N_{ij}
		double [][] N = new double[dim][dim];
		for (int j = 0; j < dim; j++) {
			for (int i = 0; i < j; i++) {
				N[i][j] = random.nextGaussian();
			}
		}
		// Build V_j
		double [] V = new double[dim];
		for (int i = 0; i < dim; i++) {
            V[i] = gammas[i].sample();
        }
		
		// Build B matrix
		double [][] B = new double[dim][dim];
		for (int j = 0; j < dim; j++) {
			double sum = 0.0;
			for (int i = 0; i < j; i++) {
				sum += Math.pow(N[i][j], 2);
			}
			B[j][j] = V[j] + sum;
		}
		for (int j = 1; j < dim; j++) {
            B[0][j] = N[0][j] * Math.sqrt(V[0]);
            B[j][0] = B[0][j];
        }
		for (int j = 1; j < dim; j++) {
            for (int i = 1; i < j; i++) {
                double sum = 0;
                for (int k = 0; k < i; k++) {
                    sum += N[k][i] * N[k][j];
                }
                B[i][j] = N[i][j] * Math.sqrt(V[i]) + sum;
                B[j][i] = B[i][j];
            }
        }
		RealMatrix BMatrix = new Array2DRowRealMatrix(B);
		RealMatrix S = cholesky.getL().multiply(BMatrix).multiply(cholesky.getLT());
		S = convertMatrixToSymmetry(S);  // To symmetric matrix
		return S;
	}
	/**To symmetric matrix
	 * @param   a matrix
	 * @return symmetric matrix
	 * ****/
    public static RealMatrix convertMatrixToSymmetry (RealMatrix c) {
    	double [][] a = c.getData();
    	for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < i; j++) {
				if (Math.abs(a[i][j] - a[j][i]) < 0.0001) {
					a[j][i] = a[i][j];
				}
			}
		}
    	RealMatrix b = new Array2DRowRealMatrix(a);
    	return b;
    } 
    
	public static double logOn2Gamma(double value) {
		return com.aliasi.util.Math.log2Gamma(value);
	}
	public static double logOn2(double value) {
		return com.aliasi.util.Math.log2(value);
	}
	public static double log2betaf(double a,double b){
		double beta = logOn2Gamma(a) + logOn2Gamma(b) - logOn2Gamma(a+b);
		return beta;
	}
	public static double exponential2(double a){
		return java.lang.Math.pow(2.0, a);
	}
}

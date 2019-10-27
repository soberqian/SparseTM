package sparseLDA;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.distribution.BetaDistribution;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.special.Gamma;

import model.Perplexity;
import util.FileUtil;
import util.FuncUtils;


/**
 * A single-sparse topic model--GibbsSampling
 * 
 * @author: qianyang
 * @email qy20115549@126.com
 */
public class SparseLDAGibbsSampling {

	public Map<String, Integer> wordToIndexMap = new HashMap<String, Integer>();;  //word to index
	public List<String> indexToWordMap = new ArrayList<String>();    //index to String word 
	public int[][] docword;
	public int M;  //document size
	public int V; // number of words in the corpus
	public int K;
	public double epsilon0;
	public double epsilon1;
	public double alpha0 ;  //超参数 alpha0 = 1E-12
	public double alpha1 ;  //超参数alpha1 = 0.1
	public double beta; 
	int[][] z;
	public int[][] ndk; // document-topic count
	public int[] ndsum; //document-topic sum
	public int[][] nkw; //topic-word count
	public int[] nksum; //topic-word sum (total number of words assigned to a topic)
	double pi_b[];  //pi参数
	boolean b[][]; //文档主题选择器
	int b_sum[];  //文档主题的个数
	JDKRandomGenerator rand; //随机数生成器
	//output
	public int topWordsOutputNumber;
	public String outputFileDirectory; 
	public String code; 
	int iterations;
	public int topWords; // number of most probable words for each topic
	public SparseLDAGibbsSampling(String inputFile, String inputFileCode, int topicNumber,
			double inputEpsilon0, double inputEpsilon1, double inputAlpha0, double inputAlpha1, 
			double inputBeta, int inputIterations, int inTopWords,
			String outputFileDir){
		//read data
		ArrayList<String> docLines = new ArrayList<String>();
		FileUtil.readLines(inputFile, docLines,inputFileCode);
		M = docLines.size();
		docword = new int[M][];
		int j = 0;
		for(String line : docLines){
			List<String> words = new ArrayList<String>();
			FileUtil.tokenizeAndLowerCase(line, words,"//s+");
			docword[j] = new int[words.size()];
			for(int i = 0; i < words.size(); i++){
				String word = words.get(i);
				if(!wordToIndexMap.containsKey(word)){
					int newIndex = wordToIndexMap.size();
					wordToIndexMap.put(word, newIndex);
					indexToWordMap.add(word);
					docword[j][i] = newIndex;
				} else {
					docword[j][i] = wordToIndexMap.get(word);
				}
			}
			j++;

		}
		V = indexToWordMap.size();
		epsilon0 = inputEpsilon0;
		epsilon1 = inputEpsilon1;
		alpha0 = inputAlpha0;
		alpha1 = inputAlpha1;
		beta = inputBeta;
		K = topicNumber;
		iterations = inputIterations;
		topWordsOutputNumber = inTopWords;
		outputFileDirectory = outputFileDir;
		code = inputFileCode;
		initialize();
	}
	//initialize the model
	public void initialize() {
		rand = new JDKRandomGenerator();
		rand.setSeed(System.currentTimeMillis());
		//贝塔分布生成
		BetaDistribution betaDist = new BetaDistribution(rand, epsilon0 , epsilon1);
		//文档d中主题k生成的单词数目
		ndk = new int[M][K];
		//文档m是否包含主题k
		b = new boolean[M][K]; 
		//文档m包含主题的数量
		b_sum = new int[M]; 
		//pi参数
		pi_b = new double[M];  
		//每篇文档包含的所有单词总数
		ndsum = new int[M];
		//主题k中单词v的数目
		nkw = new int[K][V];
		//主题k对应的单词总数
		nksum = new int[K];
		//每篇文档单词对应的主题
		z = new int[M][];
		//对每一个文档抽取pi，随机抽取值
		for (int d = 0; d < M; d++) {
			pi_b[d] = betaDist.sample();
			//刚开始初始化文档包含所有主题
			for (int k = 0; k < K; k++) {
				b[d][k] = true;
			}
			//文档包含的主题数目总和
			b_sum[d] = K;
		}

		// assign topics
		for (int d = 0; d < M; d++) {
			// words
			int Nd = docword[d].length;
			z[d] = new int[Nd];
			//循环每一个单词
			for (int n = 0; n < Nd; n++) {
				//随机赋值主题
				int topic = (int) (Math.random() * K);
				z[d][n] = topic;
				updateCount(d, z[d][n], docword[d][n], +1);
			}
		}
	}
	public void MCMCSampling() {
		//抽样更新
		for (int i = 0; i < this.iterations; i++) {
			System.out.println("iteration : " + i);
			gibbsOneIteration();  //执行gibbs采样
			//稀疏度计算
			if(i % 1 == 0) {
				//抽取二元矩阵
				sampleBinaryAMatrix();
			}
//			System.out.println(ndsum[0] + "\t" + ndsum[2]);
		}
		System.out.println("write topic word ..." );
		writeTopWordsWithProbability();
		//writeTopWords();
		System.out.println("write document topic ..." );
		writeDocumentTopic();
		System.out.println("write perplexity ..." );
		writePerplexity();
		System.out.println("write sparsity ratio ..." );
		writeSparsityRatio();
		
	}
	//gibbs采样
	public void gibbsOneIteration() {
		//sample topic
		for (int d = 0; d < docword.length; d++) {
			//循环所有单词
			for (int n = 0; n < z[d].length; n++) {
				int topic = sampleFullConditional(d, n);
				//先抽取主题
				z[d][n] = topic;
				
			}
		}
	}
	//针对不为背景词的单词抽取主题
	int sampleFullConditional(int d, int n) {
		//获取原对应的主题
		int topic = z[d][n];
		updateCount(d, z[d][n], docword[d][n], -1);
		//概率
		double[] p = new double[K];
		//循环每个主题
		for (int k = 0; k < K; k++) {
			int x = b[d][k] ? 1 : 0;
			p[k] = (ndk[d][k] + x*alpha1 + alpha0) / (ndsum[d] + K * alpha0) * (nkw[k][docword[d][n]] + beta)
					/ (nksum[k] + V * beta);
		}
		//轮盘赌抽取新主题
		topic = sample(p);
		updateCount(d, topic, docword[d][n], +1);
		//返回主题
		return topic;

	}
	//轮盘赌
	int sample(double[] p) {

		int topic = 0;
		for (int k = 1; k < p.length; k++) {
			p[k] += p[k - 1];
		}
		double u = Math.random() * p[p.length - 1];
		for (int t = 0; t < p.length; t++) {
			if (u < p[t]) {
				topic = t;
				break;
			}
		}
		return topic;
	}
	//更新统计
	void updateCount(int d, int topic, int word, int flag) {
		//文档 d中的主题topic对应的单词数目加1
		ndk[d][topic] += flag;
		//文档d
		ndsum[d] += flag;
		//主题topic对应的单词word数量加1
		nkw[topic][word] += flag;
		//主题topic对应的单词总数加1
		nksum[topic] += flag;
	}
	//抽取文档主题选择器
	public void sampleBinaryAMatrix() {
		int GIBBS_ITER = 1;
		//文档选择主题的个数
		b_sum = new int[M];
		//循环每一篇文档
		for (int m = 0; m != M; m++) {
			//循环每个主题
			for (int k = 0; k != K; k++) {
				//判断文档是否已经有该主题了,如果有则为true
				b[m][k] = (ndk[m][k]) > 0;
				//文档m包含的主题个数+1
				b_sum[m] += b[m][k] ? 1 : 0;
			}
		}
		//
		double log_diff, ratio, p;
		for (int iter = 0; iter != GIBBS_ITER; iter++) {
			for (int m = 0; m != M; m++) {
				for (int k = 0; k != K; k++) {
					if (b[m][k] && (ndk[m][k])  == 0) {
						log_diff = Gamma.logGamma(b_sum[m]*alpha1 + K*alpha0)
								- Gamma.logGamma((b_sum[m]-1)*alpha1 + K*alpha0);
						log_diff -= Gamma.logGamma(ndsum[m] + b_sum[m]*alpha1 + K*alpha0)
								- Gamma.logGamma(ndsum[m] + (b_sum[m]-1)*alpha1 + K*alpha0);

						ratio = Math.exp(log_diff) * pi_b[m] / (1.0-pi_b[m]);
						p = ratio / (1.0 + ratio);
						if (rand.nextDouble() > p) { 
							b[m][k] = false;
							b_sum[m] --;
						}
					} else if (!b[m][k]) {
						log_diff = Gamma.logGamma((b_sum[m]+1)*alpha1 + K*alpha0)
								- Gamma.logGamma(b_sum[m]*alpha1 + K*alpha0);
						log_diff -= Gamma.logGamma(ndsum[m] + (b_sum[m]+1)*alpha1 + K*alpha0)
								- Gamma.logGamma(ndsum[m] + b_sum[m]*alpha1 + K*alpha0);

						ratio = Math.exp(log_diff) * pi_b[m] / (1.0-pi_b[m]);
						p = ratio / (1.0 + ratio);
						if (rand.nextDouble() < p) { 
							b[m][k] = true;
							b_sum[m] ++;
						}
					}
				}

				BetaDistribution betaDist = new BetaDistribution(rand, epsilon0 + b_sum[m], epsilon1 + K - b_sum[m]);
				pi_b[m] = betaDist.sample();
			}
		}
	}
	//估计Theta
	public double[][] estimateTheta() {
		double[][] theta = new double[docword.length][K];
		for (int d = 0; d < docword.length; d++) {
			for (int k = 0; k < K; k++) {
				int x = b[d][k] ? 1 : 0;
				theta[d][k] = (ndk[d][k] + + x*alpha1 + alpha0) / (ndsum[d] + + b_sum[d]*alpha1 + K * alpha0);
			}
		}
		return theta;
	}
	//估计Phi
	public double[][] estimatePhi() {
		double[][] phi = new double[K][V];
		for (int k = 0; k < K; k++) {
			for (int w = 0; w < V; w++) {
				phi[k][w] = (nkw[k][w] + beta) / (nksum[k] + V * beta);
			}
		}
		return phi;
	}
	/**
	 * write top words with probability for each topic
	 */
	public void writeTopWordsWithProbability(){
		StringBuilder sBuilder = new StringBuilder();
		double[][] phi = estimatePhi();
		int topicNumber = 1;
		for (double[] phi_z : phi) {
			sBuilder.append("Topic:" + topicNumber + "\n");
			for (int i = 0; i < topWordsOutputNumber; i++) {
				int max_index = FuncUtils.maxValueIndex(phi_z);
				sBuilder.append(indexToWordMap.get(max_index) + " :" + phi_z[max_index] + "\n");
				phi_z[max_index] = 0;
			}
			sBuilder.append("\n");
			topicNumber++;
		}
		try {
			FileUtil.writeFile(outputFileDirectory + "SparseLDA_topic_word_" + K + ".txt", sBuilder.toString(),code);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	/**
	 * write theta for each document
	 */
	public void writeDocumentTopic(){
		double[][] theta = estimateTheta();
		StringBuilder sBuilder = new StringBuilder();
		for (int d = 0; d < theta.length; d++) {
			StringBuilder doc = new StringBuilder();
			for (int k = 0; k < theta[d].length; k++) {
				doc.append(theta[d][k] + "\t");
			}
			sBuilder.append(doc.toString().trim() + "\n");
		}
		try {
			FileUtil.writeFile(outputFileDirectory + "SparseLDA_doc_topic" + K + ".txt", sBuilder.toString(),code);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	/**
	 * write theta for each document
	 */
	public void writePerplexity(){
		StringBuilder sBuilder = new StringBuilder();
		double[][] theta = estimateTheta();
		double[][] phi = estimatePhi();
		double perplexity = Perplexity.lda_training_perplexity(docword, theta, phi);
		sBuilder.append(K + "\t Perplexity is: \n");
		sBuilder.append(perplexity);
		try {
			FileUtil.writeFile(outputFileDirectory + "SparseLDA_perplexity" + K + ".txt", sBuilder.toString(),code);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	/**
	 * write Sparsity Ratio for each document
	 */
	public void writeSparsityRatio(){
		StringBuilder sBuilder = new StringBuilder();
		double[] sparsityratio = estimateSparsityRatio();
		double averSparsityRatio = estimateAverSparsityRatio();
		for (int d = 0; d < sparsityratio.length; d++) {
			sBuilder.append(sparsityratio[d] + "\n");
		}
		sBuilder.append("average value:" + averSparsityRatio);
		try {
			FileUtil.writeFile(outputFileDirectory + "SparseLDA_sparsityRatio" + K + ".txt", sBuilder.toString(),code);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	public double[] estimateSparsityRatio() {
		double[] sparsityratio = new double[M];
		for (int d = 0; d < M; d++) {
			double ratio_m = 0.0;
			for (int k = 0; k < K; k++) {
				int x = b[d][k] ? 1 : 0;
				ratio_m += x;
			}
			sparsityratio[d] = 1 - ratio_m/K;
		}
		return sparsityratio;
	}
	//估计sparsity ratio
	public double estimateAverSparsityRatio() {
		double aver_sparsityratio = 0.0;
		for (int d = 0; d < M; d++) {
			aver_sparsityratio +=  (1 - b_sum[d]/K);
		}
		return aver_sparsityratio/M;
	}
	public static void main(String args[]) throws Exception{
		
		SparseLDAGibbsSampling slda = new SparseLDAGibbsSampling("datatest/shortdoc.txt", "gbk",20, 0.1, 0.1, 1E-12, 0.1, 
				0.01,2000, 100, "results/");
		slda.MCMCSampling();
	}
}

package util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class FileUtil {
	//read a file to list
	public static void readLines(String file, ArrayList<String> lines, String code) {
		BufferedReader reader = null;
		try {
			reader = new BufferedReader( new InputStreamReader( new FileInputStream( new File(file)),code));
			String line = null;
			while ((line = reader.readLine()) != null) {
				lines.add(line);
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (reader != null) {
				try {
					reader.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}

	}
	// write list to a file
	public static void writeLines(String file, ArrayList<?> counts, String code) {
		BufferedWriter writer = null;
		try {
			writer = new BufferedWriter( new OutputStreamWriter( new FileOutputStream( new File(file)),code));
			for (int i = 0; i < counts.size(); i++) {
				writer.write(counts.get(i) + "\n");
			}

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (writer != null) {
				try {
					writer.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}

	}
	/**
	 * 
	 * @param file
	 * @param content
	 * @param code
	 * @throws IOException
	 */
	public static void writeFile(String file, String content,String code) throws IOException {

		File fileOutput = new File(file);
		OutputStream out = new FileOutputStream(fileOutput, false);
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(out, code));
		bw.write(content);
		bw.close();
		out.close();
	}
	/**
	 * 
	 * @param fileInput
	 * return parent directory
	 */
	public static String getParentofFile(String fileInput) {
		File file = new File(fileInput);
		return file.getParent();
	}
	//词数组转字符串
	public static String getRecombine(ArrayList<String> wordslist){
		String text = "";
		for (int i = 0; i < wordslist.size(); i++) {
			text +=  wordslist.get(i) + " ";
		}
		text = text.trim();
		return text;
	}

	/**
	 * Judge whether a word is noise word
	 * @param string: an english word
	 * */
	public static boolean isNoiseWord(String string) {
		string = string.toLowerCase().trim();
		Pattern MY_PATTERN = Pattern.compile(".*[a-zA-Z]+.*");
		Matcher m = MY_PATTERN.matcher(string);
		// filter @xxx and URL
		if(string.matches(".*www\\..*") || string.matches(".*\\.com.*") || 
				string.matches(".*http:.*") )
			return true;
		if (!m.matches()) {
			return true;
		} else
			return false;
	}
	//判断是否为标点符号
	public static boolean isPunctuation(String str)  {
		String regEx = "[`~☆★!@#$%^&*()+=|{}':;,\\[\\]》·.<>/?~！@#￥%……（）——+|{}【】‘；：”“’。，、？]";
		Pattern p = Pattern.compile(regEx);
		Matcher m = p.matcher(str);
		if (str.length()==1 && m.matches()) {
			return true;
		}else {
			return false;
		}
	}
	/**
	 * split the document
	 * @param line
	 * @param tokens
	 * */
	public static void tokenizeAndLowerCase(String line, List<String> tokens) {
		StringTokenizer strTok = new StringTokenizer(line);
		while (strTok.hasMoreTokens()) {
			String token = strTok.nextToken();
			tokens.add(token.toLowerCase().trim());
		}
	}
	/**
	 * split the document
	 * @param line
	 * @param tokens
	 * */
	public static void tokenizeAndLowerCase(String line, List<String> tokens, String regex) {
		String[] arr = line.split(regex);
		for (int i = 0; i < arr.length; i++) {
			tokens.add(arr[i].trim());
		}
	}
	/**
	 * split the document to sentence
	 * Sentence LDA
	 * @param line
	 * @param sentences
	 * @param separator
	 * */
	public static void splitToSentence(String line, List<String> sentences, String Separator) {
		String[] sentenceArr = line.split(Separator);
		for (int i = 0; i < sentenceArr.length; i++) {
			sentences.add(sentenceArr[i]);
		}
	}
	/**
	 * translate into links for link LDA
	 * @param line
	 * @param links
	 * */
	public static void tokenizeEntity(String line, List<String> links) {
		String[] e = line.split("--");
		for (int i = 0; i < e.length; i++) {
			links.add(e[i]);
		}
	}
}

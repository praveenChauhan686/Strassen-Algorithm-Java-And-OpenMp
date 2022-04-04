package com.mat.multiplication.strassen;

import java.util.*;

class StrassenThread {
	public static void main(String[] args) {
		StrassenMatrixMultiplicationAlgorithm MMA = new StrassenMatrixMultiplicationAlgorithm();
		
		//User input for the size of the matrix
		@SuppressWarnings("resource")
		Scanner sc = new Scanner(System.in);
		System.out.println("Enter the size of matrix to be formed");
		int n = sc.nextInt();

		int i = 0;
		int j = 0;
		int k = 0;
		for (k = 0; k < 1; k++)
			try {
				{
//					int minimum = 1;
//					int maximum = 50;
					float[][] a = new float[n][n];
					float[][] b = new float[n][n];
					@SuppressWarnings("unused")
					float[][] c = new float[n][n];
					// float [][]d = new float[n][n];
					for (i = 0; i < n; i++) {
						for (j = 0; j < n; j++) {
							// a[i][j] = (float)(minimum + (int)(Math.random() * maximum));
							// b[i][j] = (float)(minimum + (int)(Math.random() * maximum));
							a[i][j] = (int) (Math.random() * 10);
						}

						for (j = 0; j < n; j++) {
							// a[i][j] = (float)(minimum + (int)(Math.random() * maximum));
							// b[i][j] = (float)(minimum + (int)(Math.random() * maximum));
							b[i][j] = (int) (Math.random() * 10);
						}
					}
					
					//Start time of the execution
					double start = System.nanoTime();
					c = MMA.strassen(a, b);
					double end = System.nanoTime();

					/*
					 * for(i=0;i<n;i++) { for(j=0;j<n;j++) { System.out.print(c[i][j]+" "); }
					 * System.out.println(); }
					 */

					//Converted the tie into seconds and printed the same after the overall execution
					System.out.println("Time taken for n = " + n + " is : " + (end - start) / 1000000000 + " Seconds");

				}
			} catch (Exception e) {
				e.printStackTrace();
			}
	}

}
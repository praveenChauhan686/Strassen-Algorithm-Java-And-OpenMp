package com.mat.multiplication.strassen;

public class StrassenSplitAddSubJoin {
	public void split(float[][] P, float[][] C, int iB, int jB) {
		int i2 = iB;
		for (int i1 = 0; i1 < C.length; i1++) {
			int j2 = jB;
			for (int j1 = 0; j1 < C.length; j1++) {
				C[i1][j1] = P[i2][j2];
				j2++;
			}
			i2++;
		}
	}

	public float[][] add(float[][] a, float[][] b) {
		int n = a.length;
		float[][] c = new float[n][n];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++)
				c[i][j] = a[i][j] + b[i][j];
		}
		return c;
	}

	public float[][] sub(float[][] a, float[][] b) {
		int n = a.length;
		float[][] c = new float[n][n];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++)
				c[i][j] = a[i][j] - b[i][j];
		}
		return c;
	}

	public void join(float[][] P, float[][] C, int iB, int jB) {
		int i2 = iB;
		for (int i1 = 0; i1 < P.length; i1++) {
			int j2 = jB;
			for (int j1 = 0; j1 < P.length; j1++) {
				C[i2][j2] = P[i1][j1];
				j2++;
			}
			i2++;
		}
	}
}

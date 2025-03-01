package org.example;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import org.apache.commons.math3.analysis.integration.IterativeLegendreGaussIntegrator;
import org.apache.commons.math3.analysis.integration.UnivariateIntegrator;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.LUDecomposition;

import javax.swing.JFrame;
import java.util.Arrays;

public class FEMSolver {
    private final double rho = 1.0;
    private final double a = 0.0;
    private final double b = 3.0;
    private final double length = Math.abs(b - a);
    private final UnivariateIntegrator integrate = new IterativeLegendreGaussIntegrator(100, 1e-6, 1e-6);

    private int e_r(double x) {
        if(x >= 0 && x <= 1) {
            return 10;
        } else if(x > 1 && x <=2) {
            return 5;
        }
        return 1;
    }

    private double shape_function(int i, double x, int n) {
        double h = length / n;
        double xi1 = h * (i - 1);
        double xi2 = h * i;
        double xi3 = h * (i + 1);

        if (x > xi1 && x <= xi2) {
            return (x - xi1) / h;
        }
        else if (x > xi2 && x <= xi3) {
            return (xi3 - x) / h;
        }

        return 0;
    }

    private double de_shape_function(int i, double x, int n) {
        double h = length / n;
        double xi1 = h * (i - 1);
        double xi2 = h * i;
        double xi3 = h * (i + 1);

        if (x > xi1 && x <= xi2) {
            return 1 / h;
        }
        else if (x > xi2 && x <= xi3) {
            return -1 / h;
        }

        return 0;
    }

    private double B(int i, int j, int n) {
        double h = length / n;
        double left = shape_function(i, 0, n) * shape_function(j, 0, n);

        double integralBoundLeft = Math.max(Math.max(a, (i - 1) * h), (j - 1) * h);
        double integralBoundRight = Math.min(Math.min((i+1) * h, (j+1) * h), b);
        if(integralBoundLeft >= integralBoundRight) {
            return 0;
        }
        double integral = integrate.integrate(Integer.MAX_VALUE, x -> de_shape_function(i, x, n) * de_shape_function(j, x, n), integralBoundLeft, integralBoundRight);

        return left - integral;
    }

    private double L(int i, int n) {
        double h = length / n;
        double left = 5 * shape_function(i, 0, n);

        double integralBoundLeft = Math.max(a, (i - 1) * h);
        double integralBoundRight = Math.min((i + 1) * h, b);
        double integral = integrate.integrate(Integer.MAX_VALUE, x -> (rho * shape_function(i, x, n)) / e_r(x), integralBoundLeft, integralBoundRight);
        return left - integral;
    }

    private double L2(int i, int n) {
        return L(i, n) - 2 * B(n, i, n);
    }

    public double[] finit_elem_method(int n) {
        RealMatrix matrixB = new Array2DRowRealMatrix(n, n);

        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                if(Math.abs(i - j) <= 1)
                    matrixB.setEntry(i, j, B(i, j, n));
            }
        }

        RealVector matrixL = new ArrayRealVector(n, 0);

        for(int i = 0; i < n; i++) {
            matrixL.setEntry(i, L2(i, n));
        }

        RealVector coefficients = new LUDecomposition(matrixB).getSolver().solve(matrixL);
        double[] result = Arrays.copyOf(coefficients.toArray(), n + 1);
        result[n] = 2;

        return result;
    }

    public void visualizeSolution(int n) {
        double[] coefficients = finit_elem_method(n);
        XYSeries series = new XYSeries("FEM Solution");

        double h = length / n;

        for (int i = 0; i <= n; i++) {
            double x = i * h;
            double value = calculateSolutionAt(x, coefficients, n);
            series.add(x, value);
        }

        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(series);

        JFreeChart chart = ChartFactory.createXYLineChart(
                "FEM Solution",
                "x",
                "u(x)",
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );

        JFrame frame = new JFrame("FEM Solution Visualization");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(new ChartPanel(chart));
        frame.pack();
        frame.setSize(800, 600);
        frame.setVisible(true);
    }

    private double calculateSolutionAt(double x, double[] coefficients, int n) {
        double sum = 0;

        // Sum up the contribution of each basis function
        for (int i = 0; i < coefficients.length-1; i++) {
            sum += coefficients[i] * shape_function(i, x, n);
        }
        sum += 2 * shape_function(n, x, n);
        return sum;
    }

    // Example usage in main method
    public static void main(String[] args) {
        int n = Integer.parseInt(args[0]);
        FEMSolver solver = new FEMSolver();
        solver.visualizeSolution(n);
    }
}
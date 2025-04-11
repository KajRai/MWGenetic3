using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace XorNeuralNetworkGA
{
    public partial class MainForm : Form
    {
        private Random random = new Random();
        private const int PopulationSize = 100;
        private const double MutationRate = 0.1;
        private const double CrossoverRate = 0.7;
        private const int MaxGenerations = 100;
        private const int TournamentSize = 3;
        private const int WeightsCount = 9; // 3 neurony po 3 wagi
        private const double WeightMin = -10.0;
        private const double WeightMax = 10.0;

        private double[][] population;
        private double[] fitness;
        private double[] xorInputs = { 0, 0, 0, 1, 1, 0, 1, 1 }; // cztery probki, dwa wejscia
        private double[] xorTargets = { 0, 1, 1, 0 }; // wyjscia

        private List<double> bestFitnessList = new List<double>();
        private List<double> avgFitnessList = new List<double>();

    }
}
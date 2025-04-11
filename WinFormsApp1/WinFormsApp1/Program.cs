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


        public MainForm()
        {
            InitializeComponent();
        }

        private void InitializeComponent()
        {
            this.btnStart = new System.Windows.Forms.Button();
            this.txtOutput = new System.Windows.Forms.RichTextBox();
            this.chart = new System.Windows.Forms.PictureBox();
            this.SuspendLayout();
            // 
            // przycisk startu
            // 
            this.btnStart.Location = new System.Drawing.Point(12, 12);
            this.btnStart.Name = "btnStart";
            this.btnStart.Size = new System.Drawing.Size(165, 30);
            this.btnStart.TabIndex = 0;
            this.btnStart.Text = "Rozpocznij algorytm";
            this.btnStart.UseVisualStyleBackColor = true;
            this.btnStart.Click += new System.EventHandler(this.btnStart_Click);
            // 
            // wypisanie
            // 
            this.txtOutput.Location = new System.Drawing.Point(12, 48);
            this.txtOutput.Name = "txtOutput";
            this.txtOutput.Size = new System.Drawing.Size(450, 250);
            this.txtOutput.TabIndex = 1;
            this.txtOutput.Text = "";
            // 
            // okreslenie wykresu
            // 
            this.chart.Location = new System.Drawing.Point(12, 310);
            this.chart.Name = "chart";
            this.chart.Size = new System.Drawing.Size(450, 250);
            this.chart.TabIndex = 2;
            this.chart.TabStop = false;
            this.chart.Paint += new System.Windows.Forms.PaintEventHandler(this.chart_Paint);
            // 
            // glowne okienko
            // 
            this.ClientSize = new System.Drawing.Size(474, 572);
            this.Controls.Add(this.chart);
            this.Controls.Add(this.txtOutput);
            this.Controls.Add(this.btnStart);
            this.Name = "MainForm";
            this.Text = "XOR Algorytm sieci neuronowej";
            this.ResumeLayout(false);
        }

        private Button btnStart;
        private RichTextBox txtOutput;
        private PictureBox chart;

        private void btnStart_Click(object sender, EventArgs e) //akcja startowa 
        {
            txtOutput.Clear();
            bestFitnessList.Clear();
            avgFitnessList.Clear();

            // inicjacja metody populacyjnej
            InitializePopulation();

            // sprawdzenie pierwszej populacji
            EvaluatePopulation();

            // sprawdzenie pierwszej populacji
            SortPopulationByFitness();

            LogGenerationInfo(0);

            // odpalenie algorytmu
            for (int generation = 1; generation <= MaxGenerations; generation++)
            {
                // nowa populacja
                double[][] newPopulation = new double[PopulationSize][];

                // najlepsze wyjscia
                newPopulation[0] = (double[])population[0].Clone();


                for (int i = 1; i < PopulationSize; i++)
                {
                    // wybranie dwoch osobnikow do turnieju
                    double[] parent1 = TournamentSelection();
                    double[] parent2 = TournamentSelection();

                    // nowe dziecko
                    double[] child;
                    if (random.NextDouble() < CrossoverRate)
                    {
                        child = Crossover(parent1, parent2);
                    }
                    else
                    {
                        child = (double[])parent1.Clone();
                    }


                    Mutate(child);

                    // dodanie nowego dziecka do populacji
                    newPopulation[i] = child;
                }


                population = newPopulation; //nowa populacja podstawiona za stara

                // powtorne sprawdzenie
                EvaluatePopulation();

                // inicjacja metody sortowania
                SortPopulationByFitness();

                // wypisanie danych co 10 - mo¿e byc za du¿o? 
                if (generation % 10 == 0 || generation == MaxGenerations)
                {
                    LogGenerationInfo(generation);
                }

                // test czy nie jestesmy w punkcie najlepszego rozwiazania
                if (fitness[0] < 0.0001)
                {
                    LogMessage($"Najlepsze rozwiazanie zostalo znalezione w generacji: {generation}");
                    break;
                }
            }

            // test najlepszego rozwiazania
            double[] bestSolution = population[0];
            TestSolution(bestSolution);

            //nowy wykres jezeli tak
            chart.Invalidate();
        }
    }
    private void InitializePopulation() //medoda do inicjalizacji populacji
    {
        population = new double[PopulationSize][];
        fitness = new double[PopulationSize];

        for (int i = 0; i < PopulationSize; i++)
        {
            population[i] = new double[WeightsCount];
            for (int j = 0; j < WeightsCount; j++)
            {
                population[i][j] = WeightMin + (WeightMax - WeightMin) * random.NextDouble();
            }
        }

        LogMessage("Populacja zainicjowana");
    }

    private void EvaluatePopulation()
    {
        double totalFitness = 0;

        for (int i = 0; i < PopulationSize; i++)
        {
            fitness[i] = CalculateFitness(population[i]);
            totalFitness += fitness[i];
        }

        // srednia dok³adnoœæ
        double avgFitness = totalFitness / PopulationSize;
        avgFitnessList.Add(avgFitness);

        // test dok³adnoœci
        double bestFitness = fitness.Min();
        bestFitnessList.Add(bestFitness);
    }

    private double CalculateFitness(double[] weights)
    {
        double sumSquaredError = 0;

        // wynik dla 4 xorów
        for (int sample = 0; sample < 4; sample++)
        {
            double input1 = xorInputs[sample * 2];
            double input2 = xorInputs[sample * 2 + 1];
            double target = xorTargets[sample];

            double output = FeedForward(input1, input2, weights);
            double error = target - output;
            sumSquaredError += error * error;
        }

        return sumSquaredError;
    }

private double FeedForward(double input1, double input2, double[] weights)
{
    // ukryte neurony do sigmoidow
    double h1 = Sigmoid(input1 * weights[0] + input2 * weights[1] + weights[2]); // peirwszy ukryty neuron
    double h2 = Sigmoid(input1 * weights[3] + input2 * weights[4] + weights[5]); // drugi neuron ukryty

    // Hidden to output layer
    double output = Sigmoid(h1 * weights[6] + h2 * weights[7] + weights[8]); // wyjscie neuronow

    return output;
}

private double Sigmoid(double x) // sigmoidy
{
    return 1.0 / (1.0 + Math.Exp(-x));
}

private void SortPopulationByFitness()
{
    // sortowanie po najlepszym dostosowaniu
    Array.Sort(fitness, population);
}

private double[] TournamentSelection()
{
   
    int[] tournamentIndices = new int[TournamentSize];
    for (int i = 0; i < TournamentSize; i++)
    {
        tournamentIndices[i] = random.Next(PopulationSize);
    }

    // najlepsze jednostki do turnieju
    int bestIndex = tournamentIndices[0];
    for (int i = 1; i < TournamentSize; i++)
    {
        if (fitness[tournamentIndices[i]] < fitness[bestIndex])
        {
            bestIndex = tournamentIndices[i];
        }
    }

    return (double[])population[bestIndex].Clone();
}

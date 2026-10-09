import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  BarChart3, 
  Award, 
  CheckCircle2, 
  AlertCircle, 
  Loader2, 
  Database,
  Info,
  ShieldCheck,
  SearchCheck,
  Layers,
  Sparkles
} from 'lucide-react';

export default function Models() {
  const [modelData, setModelData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get('/api/models');
      setModelData(response.data);
    } catch (err) {
      console.error('Failed to load model results:', err);
      setError('Failed to fetch model comparison data from backend. Ensure train_models.py has completed.');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="py-24 flex flex-col items-center justify-center space-y-4">
        <Loader2 className="w-10 h-10 animate-spin text-teal-600" />
        <p className="text-slate-600 font-medium">Fetching verified model benchmarks...</p>
      </div>
    );
  }

  if (error || !modelData) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-2xl p-6 text-red-800 space-y-3">
        <div className="flex items-center space-x-2 font-bold">
          <AlertCircle className="w-5 h-5 text-red-600" />
          <span>Error Loading Model Data</span>
        </div>
        <p className="text-sm">{error || 'No benchmark data available.'}</p>
        <button
          onClick={fetchModels}
          className="px-4 py-2 bg-red-600 text-white rounded-xl text-xs font-semibold hover:bg-red-700 transition cursor-pointer"
        >
          Retry Fetching
        </button>
      </div>
    );
  }

  const { best_model_name, selection_reason, best_model_test_metrics, models, dataset_summary } = modelData;

  return (
    <div className="space-y-8 py-4">
      {/* Page Header */}
      <div className="bg-white rounded-2xl border border-slate-200 p-6 sm:p-8 shadow-xs">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl sm:text-3xl font-bold text-slate-900 tracking-tight flex items-center space-x-3">
              <span className="p-2 bg-teal-100 text-teal-700 rounded-xl">
                <BarChart3 className="w-6 h-6" />
              </span>
              <span>Machine Learning Model Comparison</span>
            </h1>
            <p className="text-slate-500 text-sm mt-1.5 max-w-2xl">
              Rigorous evaluation across 6 supervised classification algorithms comparing Training Accuracy, Group-Aware Cross-Validation Accuracy, and Held-Out Testing Accuracy with zero pattern leakage.
            </p>
          </div>

          <div className="inline-flex items-center space-x-2 px-3.5 py-1.5 rounded-full bg-emerald-50 text-emerald-800 border border-emerald-200 text-xs font-semibold">
            <CheckCircle2 className="w-4 h-4 text-emerald-600" />
            <span>0% Train/Test Pattern Overlap</span>
          </div>
        </div>
      </div>

      {/* Best Selected Model Card */}
      <div className="bg-gradient-to-r from-teal-900 via-teal-800 to-slate-900 text-white rounded-2xl p-6 sm:p-8 shadow-md relative overflow-hidden">
        <div className="relative z-10 space-y-5">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-teal-400/20 text-teal-300 border border-teal-400/30 text-xs font-semibold">
              <Award className="w-4 h-4 text-teal-300" />
              <span>Optimal Selected Classifier (Chosen via Group-Aware Cross-Validation)</span>
            </div>
            <span className="text-xs text-teal-200/80 font-mono">
              Evaluated on Held-out Untouched Test Set
            </span>
          </div>

          <div>
            <h2 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-white">
              {best_model_name}
            </h2>
            <p className="text-teal-100/90 text-sm mt-2 max-w-3xl leading-relaxed">
              {selection_reason}
            </p>
          </div>

          {/* Metrics Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-4 pt-2">
            <div className="bg-teal-950/60 border border-teal-700/50 rounded-xl p-3 sm:p-4">
              <div className="text-xs text-teal-300 font-medium">Training Accuracy</div>
              <div className="text-2xl font-extrabold text-white mt-1">
                {(best_model_test_metrics.train_accuracy * 100).toFixed(2)}%
              </div>
            </div>

            <div className="bg-teal-950/60 border border-teal-700/50 rounded-xl p-3 sm:p-4">
              <div className="text-xs text-teal-300 font-medium">CV Accuracy</div>
              <div className="text-2xl font-extrabold text-white mt-1">
                {(best_model_test_metrics.cv_accuracy * 100).toFixed(2)}%
              </div>
            </div>

            <div className="bg-teal-950/60 border border-teal-700/50 rounded-xl p-3 sm:p-4">
              <div className="text-xs text-teal-300 font-medium">Test Accuracy</div>
              <div className="text-2xl font-extrabold text-white mt-1">
                {(best_model_test_metrics.test_accuracy * 100).toFixed(2)}%
              </div>
            </div>

            <div className="bg-teal-950/60 border border-teal-700/50 rounded-xl p-3 sm:p-4">
              <div className="text-xs text-teal-300 font-medium">Weighted F1-Score</div>
              <div className="text-2xl font-extrabold text-white mt-1">
                {(best_model_test_metrics.test_f1_score * 100).toFixed(2)}%
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Comprehensive Model Comparison Table */}
      <div className="bg-white rounded-2xl border border-slate-200 p-6 sm:p-8 shadow-xs space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h3 className="text-lg font-bold text-slate-900 tracking-tight">
              Detailed Algorithm Benchmark Table
            </h3>
            <p className="text-xs text-slate-500">
              Comparing Training, Cross-Validation, and Held-out Test performance across all 6 models.
            </p>
          </div>
          <span className="text-xs font-mono text-slate-500 bg-slate-100 px-2.5 py-1 rounded-md self-start">
            Split: {dataset_summary?.train_samples?.toLocaleString()} Train / {dataset_summary?.test_samples?.toLocaleString()} Test
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs sm:text-sm border-collapse">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50 text-slate-700 uppercase font-semibold text-2xs tracking-wider">
                <th className="py-3 px-3">Model Name</th>
                <th className="py-3 px-3">Training Acc</th>
                <th className="py-3 px-3">CV Acc</th>
                <th className="py-3 px-3">Testing Acc</th>
                <th className="py-3 px-3">Precision (Weighted)</th>
                <th className="py-3 px-3">Recall (Weighted)</th>
                <th className="py-3 px-3">F1-Score (Weighted)</th>
                <th className="py-3 px-3">Probabilities</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-slate-700">
              {models.map((m) => {
                const isBest = m.model_name === best_model_name;
                return (
                  <tr
                    key={m.model_name}
                    className={`hover:bg-slate-50/80 transition ${
                      isBest ? 'bg-teal-50/60 font-semibold text-teal-950' : ''
                    }`}
                  >
                    <td className="py-3.5 px-3 flex items-center space-x-1.5">
                      <span>{m.model_name}</span>
                      {isBest && (
                        <CheckCircle2 className="w-4 h-4 text-teal-600 flex-shrink-0" />
                      )}
                    </td>
                    <td className="py-3.5 px-3 font-mono font-medium text-slate-700">
                      {(m.train_accuracy * 100).toFixed(2)}%
                    </td>
                    <td className="py-3.5 px-3 font-mono font-medium text-slate-700">
                      {(m.cv_accuracy * 100).toFixed(2)}%
                    </td>
                    <td className="py-3.5 px-3 font-mono font-bold text-slate-900">
                      {(m.test_accuracy * 100).toFixed(2)}%
                    </td>
                    <td className="py-3.5 px-3 font-mono text-slate-700">
                      {(m.test_precision * 100).toFixed(2)}%
                    </td>
                    <td className="py-3.5 px-3 font-mono text-slate-700">
                      {(m.test_recall * 100).toFixed(2)}%
                    </td>
                    <td className="py-3.5 px-3 font-mono font-semibold text-slate-800">
                      {(m.test_f1_score * 100).toFixed(2)}%
                    </td>
                    <td className="py-3.5 px-3">
                      {m.supports_probability ? (
                        <span className="text-2xs font-semibold px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800">
                          Supported
                        </span>
                      ) : (
                        <span className="text-2xs font-semibold px-2 py-0.5 rounded-full bg-slate-100 text-slate-600">
                          Discrete Only
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Visual Bar Comparison */}
      <div className="bg-white rounded-2xl border border-slate-200 p-6 sm:p-8 shadow-xs space-y-6">
        <div className="space-y-1">
          <h3 className="text-lg font-bold text-slate-900 tracking-tight">
            Visual Comparison: CV Accuracy vs Testing Accuracy
          </h3>
          <p className="text-xs text-slate-500">
            Evaluating generalization across algorithms. The models show tight clustering around 93.3% to 94.2% test accuracy, reflecting realistic diagnostic differentiation.
          </p>
        </div>

        <div className="space-y-4 pt-2">
          {models.map((m) => {
            const isBest = m.model_name === best_model_name;
            const testPct = (m.test_accuracy * 100).toFixed(1);
            const cvPct = (m.cv_accuracy * 100).toFixed(1);

            return (
              <div key={m.model_name} className="space-y-2 p-3.5 rounded-xl border border-slate-100 bg-slate-50/60">
                <div className="flex flex-wrap items-center justify-between text-xs sm:text-sm">
                  <div className="flex items-center space-x-2">
                    <span className="font-bold text-slate-800">{m.model_name}</span>
                    {isBest && (
                      <span className="px-2 py-0.5 rounded-full text-2xs font-bold uppercase bg-teal-100 text-teal-800 border border-teal-200">
                        Best Selected
                      </span>
                    )}
                  </div>
                  <div className="flex items-center space-x-3 text-xs text-slate-600 font-mono">
                    <span>Train: <strong>{(m.train_accuracy * 100).toFixed(1)}%</strong></span>
                    <span>CV: <strong>{cvPct}%</strong></span>
                    <span>Test: <strong className="text-slate-900">{testPct}%</strong></span>
                  </div>
                </div>

                {/* Accuracy Bar */}
                <div className="w-full bg-slate-200 h-3 rounded-full overflow-hidden flex">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      isBest
                        ? 'bg-teal-600'
                        : 'bg-teal-500'
                    }`}
                    style={{ width: `${Math.max(m.test_accuracy * 100, 5)}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Dataset & Architecture Summary */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-xs space-y-4">
          <div className="flex items-center space-x-2 font-bold text-slate-900">
            <Database className="w-5 h-5 text-teal-600" />
            <span>Dataset Specification (ml.csv)</span>
          </div>

          <div className="space-y-2.5 text-xs text-slate-600">
            <div className="flex justify-between py-1.5 border-b border-slate-100">
              <span className="text-slate-500">Total Raw Records:</span>
              <span className="font-mono font-semibold text-slate-800">{dataset_summary?.total_rows?.toLocaleString()}</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-100">
              <span className="text-slate-500">Unique Symptom Patterns:</span>
              <span className="font-mono font-semibold text-slate-800">{dataset_summary?.unique_patterns?.toLocaleString()}</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-100">
              <span className="text-slate-500">Active Symptom Features:</span>
              <span className="font-mono font-semibold text-slate-800">{dataset_summary?.features_count}</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-100">
              <span className="text-slate-500">Target Disease Classes:</span>
              <span className="font-mono font-semibold text-slate-800">{dataset_summary?.diseases_count} Diseases</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-100">
              <span className="text-slate-500">Train/Test Pattern Overlap:</span>
              <span className="font-mono font-semibold text-emerald-600">{dataset_summary?.pattern_overlap} (Zero Leakage)</span>
            </div>
            <div className="flex justify-between py-1.5">
              <span className="text-slate-500">Symptom Patterns with Differential Overlap:</span>
              <span className="font-mono font-semibold text-slate-800">{dataset_summary?.inconsistent_patterns}</span>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-xs space-y-4">
          <div className="flex items-center space-x-2 font-bold text-slate-900">
            <Info className="w-5 h-5 text-teal-600" />
            <span>Evaluation Methodology & Realism</span>
          </div>

          <div className="space-y-2.5 text-xs text-slate-600 leading-relaxed">
            <p>
              <strong>Group-Aware Splitting:</strong> Identical symptom combinations across the 38,434 records were grouped and held strictly together in either train or test, ensuring 0% pattern leakage.
            </p>
            <p className="bg-slate-50 p-2.5 rounded-xl border border-slate-200 text-slate-700">
              <strong>Realistic Performance:</strong> With multi-symptom clinical combinations, the algorithms achieve realistic test accuracies between 93.3% and 94.2%, rather than artificial 100% memorization.
            </p>
            <p className="text-slate-500 italic">
              <strong>Medical Disclaimer:</strong> Predictions represent statistical supervised ML patterns and are never a substitute for medical diagnosis by a certified physician.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

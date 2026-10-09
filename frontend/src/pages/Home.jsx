import React from 'react';
import { Stethoscope, ArrowRight, CheckCircle, Database, Cpu, ShieldCheck, Activity, Layers } from 'lucide-react';

export default function Home({ setActiveTab, backendStatus }) {
  return (
    <div className="space-y-12 py-6">
      {/* Hero Section */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-teal-700 via-teal-800 to-slate-900 text-white p-8 sm:p-12 shadow-md">
        <div className="relative z-10 max-w-3xl space-y-6">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-teal-500/20 text-teal-200 border border-teal-400/30 text-xs font-semibold uppercase tracking-wider">
            <Activity className="w-3.5 h-3.5" />
            <span>Supervised Multiclass ML System</span>
          </div>

          <h1 className="text-3xl sm:text-5xl font-extrabold tracking-tight leading-tight">
            AI-Based Disease Prediction System
          </h1>

          <p className="text-base sm:text-lg text-teal-100/90 leading-relaxed max-w-2xl">
            A complete, college-grade Machine Learning solution designed to evaluate patient-reported symptoms
            and infer potential diseases using locally trained classification algorithms.
          </p>

          <div className="flex flex-wrap items-center gap-4 pt-2">
            <button
              onClick={() => setActiveTab('predict')}
              className="inline-flex items-center space-x-2 bg-white text-teal-900 hover:bg-teal-50 px-6 py-3.5 rounded-xl font-bold shadow-sm transition transform hover:-translate-y-0.5 cursor-pointer"
            >
              <Stethoscope className="w-5 h-5 text-teal-700" />
              <span>Start Prediction</span>
              <ArrowRight className="w-4 h-4 ml-1" />
            </button>

            <button
              onClick={() => setActiveTab('models')}
              className="inline-flex items-center space-x-2 bg-teal-900/60 hover:bg-teal-900 text-teal-100 border border-teal-600/40 px-5 py-3.5 rounded-xl font-medium transition cursor-pointer"
            >
              <Cpu className="w-4 h-4 text-teal-300" />
              <span>Explore Model Benchmarks</span>
            </button>
          </div>
        </div>

        {/* Decorative background grid */}
        <div className="absolute -right-10 -bottom-10 w-96 h-96 bg-teal-500/10 rounded-full blur-3xl pointer-events-none" />
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 sm:gap-6">
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs flex items-center space-x-4">
          <div className="w-12 h-12 rounded-lg bg-teal-50 text-teal-700 flex items-center justify-center flex-shrink-0">
            <Layers className="w-6 h-6" />
          </div>
          <div>
            <div className="text-2xl font-bold text-slate-900">
              {backendStatus?.total_features || 132}
            </div>
            <div className="text-xs text-slate-500 font-medium">Binary Symptoms</div>
          </div>
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs flex items-center space-x-4">
          <div className="w-12 h-12 rounded-lg bg-cyan-50 text-cyan-700 flex items-center justify-center flex-shrink-0">
            <Database className="w-6 h-6" />
          </div>
          <div>
            <div className="text-2xl font-bold text-slate-900">
              {backendStatus?.total_diseases || 41}
            </div>
            <div className="text-xs text-slate-500 font-medium">Target Disease Classes</div>
          </div>
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs flex items-center space-x-4">
          <div className="w-12 h-12 rounded-lg bg-indigo-50 text-indigo-700 flex items-center justify-center flex-shrink-0">
            <Cpu className="w-6 h-6" />
          </div>
          <div>
            <div className="text-2xl font-bold text-slate-900">6</div>
            <div className="text-xs text-slate-500 font-medium">Evaluated Classifiers</div>
          </div>
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs flex items-center space-x-4">
          <div className="w-12 h-12 rounded-lg bg-emerald-50 text-emerald-700 flex items-center justify-center flex-shrink-0">
            <ShieldCheck className="w-6 h-6" />
          </div>
          <div>
            <div className="text-2xl font-bold text-slate-900">100%</div>
            <div className="text-xs text-slate-500 font-medium">Local & Private ML</div>
          </div>
        </div>
      </div>

      {/* How the System Works */}
      <div className="bg-white rounded-2xl border border-slate-200 p-8 shadow-xs space-y-6">
        <div className="space-y-1">
          <h2 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">
            How The System Works
          </h2>
          <p className="text-sm text-slate-500">
            A streamlined 4-step workflow from user symptom selection to machine learning disease probability estimation.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 pt-2">
          {/* Step 1 */}
          <div className="border border-slate-200 rounded-xl p-5 bg-slate-50/50 space-y-3 relative hover:border-teal-300 transition">
            <div className="w-8 h-8 rounded-full bg-teal-600 text-white font-bold flex items-center justify-center text-sm shadow-xs">
              1
            </div>
            <h3 className="font-semibold text-slate-900">Symptom Selection</h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              Users search and choose one or more symptoms from the dynamic list loaded directly from the training dataset.
            </p>
          </div>

          {/* Step 2 */}
          <div className="border border-slate-200 rounded-xl p-5 bg-slate-50/50 space-y-3 relative hover:border-teal-300 transition">
            <div className="w-8 h-8 rounded-full bg-teal-600 text-white font-bold flex items-center justify-center text-sm shadow-xs">
              2
            </div>
            <h3 className="font-semibold text-slate-900">Feature Encoding</h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              The backend constructs an exact 132-dimensional binary vector (1 for present symptoms, 0 otherwise) matching training order.
            </p>
          </div>

          {/* Step 3 */}
          <div className="border border-slate-200 rounded-xl p-5 bg-slate-50/50 space-y-3 relative hover:border-teal-300 transition">
            <div className="w-8 h-8 rounded-full bg-teal-600 text-white font-bold flex items-center justify-center text-sm shadow-xs">
              3
            </div>
            <h3 className="font-semibold text-slate-900">Model Inference</h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              The best-performing model (Random Forest) evaluates the symptom vector and computes multi-class classification scores.
            </p>
          </div>

          {/* Step 4 */}
          <div className="border border-slate-200 rounded-xl p-5 bg-slate-50/50 space-y-3 relative hover:border-teal-300 transition">
            <div className="w-8 h-8 rounded-full bg-teal-600 text-white font-bold flex items-center justify-center text-sm shadow-xs">
              4
            </div>
            <h3 className="font-semibold text-slate-900">Result & History</h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              The primary prediction and top 3 differential candidates are displayed, accompanied by medical disclaimers and SQLite logging.
            </p>
          </div>
        </div>
      </div>

      {/* Project Architecture Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-xs space-y-4">
          <h3 className="font-bold text-slate-900 flex items-center space-x-2">
            <CheckCircle className="w-5 h-5 text-teal-600" />
            <span>Key Engineering Highlights</span>
          </h3>
          <ul className="space-y-2.5 text-xs sm:text-sm text-slate-600">
            <li className="flex items-start space-x-2">
              <span className="text-teal-600 font-bold">•</span>
              <span><strong>Leakage-Free Splitting:</strong> Deduplicated symptom patterns ensure 0% pattern overlap between training and testing.</span>
            </li>
            <li className="flex items-start space-x-2">
              <span className="text-teal-600 font-bold">•</span>
              <span><strong>Zero Constant Noise:</strong> Columns with zero variance (<code className="text-xs bg-slate-100 px-1 py-0.5 rounded">fluid_overload</code>, <code className="text-xs bg-slate-100 px-1 py-0.5 rounded">Urinating_a_lot</code>) removed.</span>
            </li>
            <li className="flex items-start space-x-2">
              <span className="text-teal-600 font-bold">•</span>
              <span><strong>Excluded Leakage Feature:</strong> The <code className="text-xs bg-slate-100 px-1 py-0.5 rounded">medicine</code> column is strictly excluded from features.</span>
            </li>
          </ul>
        </div>

        <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-xs space-y-4">
          <h3 className="font-bold text-slate-900 flex items-center space-x-2">
            <ShieldCheck className="w-5 h-5 text-teal-600" />
            <span>Ethical Healthcare AI Standards</span>
          </h3>
          <ul className="space-y-2.5 text-xs sm:text-sm text-slate-600">
            <li className="flex items-start space-x-2">
              <span className="text-teal-600 font-bold">•</span>
              <span><strong>No Medication Prescriptions:</strong> Does not recommend prescription pharmaceuticals.</span>
            </li>
            <li className="flex items-start space-x-2">
              <span className="text-teal-600 font-bold">•</span>
              <span><strong>Prominent Disclaimers:</strong> Emphasizes clearly that predictions do not constitute a medical diagnosis.</span>
            </li>
            <li className="flex items-start space-x-2">
              <span className="text-teal-600 font-bold">•</span>
              <span><strong>Offline & Private:</strong> Runs locally with SQLite and Scikit-learn without transmitting data to 3rd-party LLMs.</span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}

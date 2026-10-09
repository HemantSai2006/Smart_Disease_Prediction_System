import React, { useState, useEffect, useMemo } from 'react';
import axios from 'axios';
import { 
  Search, 
  X, 
  Check, 
  AlertCircle, 
  Loader2, 
  Stethoscope, 
  Sparkles, 
  RefreshCw, 
  Info, 
  ShieldAlert,
  ChevronRight,
  TrendingUp
} from 'lucide-react';

export default function Predict() {
  const [symptomsList, setSymptomsList] = useState([]);
  const [selectedSymptoms, setSelectedSymptoms] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [loadingSymptoms, setLoadingSymptoms] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Fetch symptoms dynamically from backend on mount
  useEffect(() => {
    fetchSymptoms();
  }, []);

  const fetchSymptoms = async () => {
    setLoadingSymptoms(true);
    setError(null);
    try {
      const response = await axios.get('/api/symptoms');
      if (response.data && response.data.symptoms) {
        setSymptomsList(response.data.symptoms);
      }
    } catch (err) {
      console.error('Failed to load symptoms:', err);
      setError('Could not connect to the backend API to fetch symptoms. Ensure the FastAPI server is running on port 8000.');
    } finally {
      setLoadingSymptoms(false);
    }
  };

  // Filter symptoms based on search input
  const filteredSymptoms = useMemo(() => {
    if (!searchQuery.trim()) return symptomsList;
    const q = searchQuery.toLowerCase();
    return symptomsList.filter(
      (s) => s.label.toLowerCase().includes(q) || s.id.toLowerCase().includes(q)
    );
  }, [symptomsList, searchQuery]);

  const toggleSymptom = (symptomId) => {
    setSelectedSymptoms((prev) =>
      prev.includes(symptomId)
        ? prev.filter((id) => id !== symptomId)
        : [...prev, symptomId]
    );
    // Clear previous prediction when selections change
    if (result) setResult(null);
  };

  const removeSymptom = (symptomId) => {
    setSelectedSymptoms((prev) => prev.filter((id) => id !== symptomId));
    if (result) setResult(null);
  };

  const clearSelection = () => {
    setSelectedSymptoms([]);
    setResult(null);
    setError(null);
  };

  const handlePredict = async () => {
    if (selectedSymptoms.length === 0) {
      setError('Please select at least one symptom before predicting.');
      return;
    }

    setPredicting(true);
    setError(null);
    try {
      const response = await axios.post('/api/predict', {
        symptoms: selectedSymptoms,
      });
      setResult(response.data);
    } catch (err) {
      console.error('Prediction failed:', err);
      const msg = err.response?.data?.detail || 'An error occurred while evaluating the symptoms.';
      setError(msg);
    } finally {
      setPredicting(false);
    }
  };

  return (
    <div className="space-y-8 py-4">
      {/* Page Title & Intro */}
      <div className="bg-white rounded-2xl border border-slate-200 p-6 sm:p-8 shadow-xs">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl sm:text-3xl font-bold text-slate-900 tracking-tight flex items-center space-x-3">
              <span className="p-2 bg-teal-100 text-teal-700 rounded-xl">
                <Stethoscope className="w-6 h-6" />
              </span>
              <span>Disease Prediction</span>
            </h1>
            <p className="text-slate-500 text-sm mt-1.5 max-w-2xl">
              Select all symptoms you are currently experiencing. Our machine learning model will analyze the symptom pattern and estimate potential disease categories.
            </p>
          </div>

          <div className="flex items-center space-x-3">
            <span className="text-xs font-semibold text-slate-500 bg-slate-100 px-3 py-1.5 rounded-lg border border-slate-200">
              {symptomsList.length} Symptoms Available
            </span>
          </div>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-4 flex items-start space-x-3 text-red-800 text-sm">
          <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="font-semibold">Attention Needed</p>
            <p className="text-red-700 text-xs mt-0.5">{error}</p>
          </div>
          <button 
            onClick={() => setError(null)}
            className="text-red-500 hover:text-red-700 text-xs font-bold"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Main Layout: 2 Columns */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Symptom Selector (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-xs space-y-5">
            {/* Search Input & Action Buttons */}
            <div className="flex flex-col sm:flex-row gap-3">
              <div className="relative flex-1">
                <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-3.5" />
                <input
                  type="text"
                  placeholder="Search symptoms (e.g. fever, headache, rash)..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-10 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-teal-500 focus:bg-white transition"
                />
                {searchQuery && (
                  <button
                    onClick={() => setSearchQuery('')}
                    className="absolute right-3 top-3 text-slate-400 hover:text-slate-600"
                  >
                    <X className="w-4 h-4" />
                  </button>
                )}
              </div>

              {selectedSymptoms.length > 0 && (
                <button
                  onClick={clearSelection}
                  className="px-4 py-2.5 text-xs font-semibold text-slate-600 hover:text-red-600 bg-slate-100 hover:bg-red-50 rounded-xl border border-slate-200 transition flex items-center justify-center space-x-1 cursor-pointer"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  <span>Clear ({selectedSymptoms.length})</span>
                </button>
              )}
            </div>

            {/* Selected Symptoms Chips */}
            {selectedSymptoms.length > 0 && (
              <div className="space-y-2 pt-1 border-t border-slate-100">
                <div className="flex items-center justify-between text-xs font-semibold text-slate-500">
                  <span>Selected Symptoms ({selectedSymptoms.length})</span>
                </div>
                <div className="flex flex-wrap gap-2 max-h-36 overflow-y-auto p-1">
                  {selectedSymptoms.map((id) => {
                    const item = symptomsList.find((s) => s.id === id);
                    const label = item ? item.label : id;
                    return (
                      <span
                        key={id}
                        className="inline-flex items-center space-x-1 bg-teal-50 text-teal-800 border border-teal-200 px-3 py-1 rounded-full text-xs font-medium shadow-2xs"
                      >
                        <span>{label}</span>
                        <button
                          onClick={() => removeSymptom(id)}
                          className="hover:text-red-600 ml-1 rounded-full p-0.5 focus:outline-none cursor-pointer"
                        >
                          <X className="w-3 h-3" />
                        </button>
                      </span>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Symptoms Checklist */}
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs font-semibold text-slate-500 px-1">
                <span>Select Symptoms from Dataset</span>
                <span>Showing {filteredSymptoms.length} items</span>
              </div>

              {loadingSymptoms ? (
                <div className="py-16 flex flex-col items-center justify-center space-y-3 text-slate-400">
                  <Loader2 className="w-8 h-8 animate-spin text-teal-600" />
                  <span className="text-sm font-medium">Loading symptoms from ML backend...</span>
                </div>
              ) : filteredSymptoms.length === 0 ? (
                <div className="py-12 text-center text-slate-400 text-sm bg-slate-50 rounded-xl border border-dashed border-slate-200">
                  No symptoms matching "{searchQuery}"
                </div>
              ) : (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-h-96 overflow-y-auto pr-1">
                  {filteredSymptoms.map((symptom) => {
                    const isSelected = selectedSymptoms.includes(symptom.id);
                    return (
                      <div
                        key={symptom.id}
                        onClick={() => toggleSymptom(symptom.id)}
                        className={`flex items-center space-x-3 p-2.5 rounded-xl border text-xs sm:text-sm font-medium transition cursor-pointer select-none ${
                          isSelected
                            ? 'bg-teal-50/80 border-teal-300 text-teal-900 shadow-2xs'
                            : 'bg-white hover:bg-slate-50 border-slate-200 text-slate-700'
                        }`}
                      >
                        <div
                          className={`w-4 h-4 rounded flex items-center justify-center border transition flex-shrink-0 ${
                            isSelected
                              ? 'bg-teal-600 border-teal-600 text-white'
                              : 'border-slate-300 bg-white'
                          }`}
                        >
                          {isSelected && <Check className="w-3 h-3" />}
                        </div>
                        <span className="truncate">{symptom.label}</span>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Predict Button */}
            <div className="pt-2">
              <button
                onClick={handlePredict}
                disabled={predicting || selectedSymptoms.length === 0}
                className={`w-full py-3.5 px-6 rounded-xl font-bold text-white shadow-sm flex items-center justify-center space-x-2 transition cursor-pointer ${
                  predicting || selectedSymptoms.length === 0
                    ? 'bg-slate-300 cursor-not-allowed opacity-75'
                    : 'bg-teal-600 hover:bg-teal-700 active:bg-teal-800'
                }`}
              >
                {predicting ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    <span>Evaluating Symptoms...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-5 h-5" />
                    <span>Predict Disease ({selectedSymptoms.length} Selected)</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Right Column: Prediction Results (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          {result ? (
            <div className="bg-white rounded-2xl border border-teal-200 p-6 shadow-sm space-y-6 animate-fadeIn">
              {/* Header Badge */}
              <div className="flex items-center justify-between pb-3 border-b border-slate-100">
                <span className="text-xs font-semibold text-teal-700 bg-teal-50 px-2.5 py-1 rounded-full border border-teal-200 flex items-center space-x-1">
                  <TrendingUp className="w-3 h-3" />
                  <span>ML Prediction Generated</span>
                </span>
                <span className="text-xs text-slate-500 font-mono">
                  Model: {result.model_name}
                </span>
              </div>

              {/* Primary Disease Card */}
              <div className="bg-gradient-to-br from-teal-50 to-emerald-50/50 rounded-xl p-5 border border-teal-200/60 space-y-2">
                <div className="text-xs font-bold text-teal-700 uppercase tracking-wider">
                  Most Likely Diagnosis
                </div>
                <div className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
                  {result.predicted_disease}
                </div>
                {result.top_predictions?.[0] && (
                  <div className="text-xs text-teal-800 font-medium">
                    Relative Statistical Score:{' '}
                    <span className="font-bold">
                      {(result.top_predictions[0].score * 100).toFixed(1)}%
                    </span>
                  </div>
                )}
              </div>

              {/* Top 3 Predictions Differential List */}
              {result.top_predictions && result.top_predictions.length > 0 && (
                <div className="space-y-3">
                  <h4 className="text-xs font-bold text-slate-600 uppercase tracking-wider">
                    Top Differential Candidates
                  </h4>
                  <div className="space-y-2.5">
                    {result.top_predictions.map((item, idx) => (
                      <div
                        key={idx}
                        className="p-3 rounded-xl border border-slate-100 bg-slate-50/70 space-y-1.5"
                      >
                        <div className="flex justify-between items-center text-xs">
                          <span className="font-bold text-slate-800">
                            {idx + 1}. {item.disease}
                          </span>
                          <span className="font-mono font-semibold text-slate-600">
                            {(item.score * 100).toFixed(1)}%
                          </span>
                        </div>
                        {/* Progress Bar */}
                        <div className="w-full bg-slate-200 h-2 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full ${
                              idx === 0
                                ? 'bg-teal-600'
                                : idx === 1
                                ? 'bg-teal-400'
                                : 'bg-slate-400'
                            }`}
                            style={{ width: `${Math.max(item.score * 100, 3)}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Disclaimer Notice */}
              <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 space-y-2 text-xs text-amber-900">
                <div className="flex items-center space-x-1.5 font-bold text-amber-800">
                  <ShieldAlert className="w-4 h-4 text-amber-600 flex-shrink-0" />
                  <span>Important Medical Disclaimer</span>
                </div>
                <p className="leading-relaxed text-amber-800/90">
                  {result.disclaimer}
                </p>
                <p className="text-amber-700/80 italic pt-1 border-t border-amber-200/50">
                  No automated prescription drugs or treatments are provided. Always seek direct medical consultation with a registered healthcare provider.
                </p>
              </div>
            </div>
          ) : (
            /* Idle Placeholder */
            <div className="bg-white rounded-2xl border border-dashed border-slate-200 p-8 text-center space-y-4">
              <div className="w-14 h-14 rounded-2xl bg-slate-100 text-slate-400 flex items-center justify-center mx-auto">
                <Stethoscope className="w-7 h-7" />
              </div>
              <div className="space-y-1">
                <h3 className="font-bold text-slate-800 text-base">Prediction Awaiting Input</h3>
                <p className="text-xs text-slate-500 max-w-xs mx-auto leading-relaxed">
                  Select one or more symptoms from the list on the left and click "Predict Disease" to view statistical predictions.
                </p>
              </div>
              <div className="inline-flex items-center text-xs text-teal-600 font-semibold space-x-1">
                <span>Quick Tip: Select 3 or more symptoms for higher specificity</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

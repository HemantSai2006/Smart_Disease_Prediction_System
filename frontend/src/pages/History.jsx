import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  History as HistoryIcon, 
  RotateCw, 
  Calendar, 
  Stethoscope, 
  AlertCircle, 
  Loader2, 
  CheckCircle2, 
  Search,
  Tag
} from 'lucide-react';

export default function History() {
  const [historyRecords, setHistoryRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filterQuery, setFilterQuery] = useState('');

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get('/api/history');
      if (response.data && response.data.history) {
        setHistoryRecords(response.data.history);
      }
    } catch (err) {
      console.error('Failed to load history:', err);
      setError('Failed to connect to backend to retrieve prediction history.');
    } finally {
      setLoading(false);
    }
  };

  const formatSymptom = (str) => {
    return str.replace('.1', '').replace(/_/g, ' ').trim();
  };

  const filtered = historyRecords.filter((rec) => {
    if (!filterQuery.trim()) return true;
    const q = filterQuery.toLowerCase();
    const diseaseMatch = rec.predicted_disease?.toLowerCase().includes(q);
    const symptomMatch = rec.symptoms?.some((s) => s.toLowerCase().includes(q));
    return diseaseMatch || symptomMatch;
  });

  return (
    <div className="space-y-8 py-4">
      {/* Header */}
      <div className="bg-white rounded-2xl border border-slate-200 p-6 sm:p-8 shadow-xs">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl sm:text-3xl font-bold text-slate-900 tracking-tight flex items-center space-x-3">
              <span className="p-2 bg-teal-100 text-teal-700 rounded-xl">
                <HistoryIcon className="w-6 h-6" />
              </span>
              <span>Prediction History</span>
            </h1>
            <p className="text-slate-500 text-sm mt-1.5 max-w-2xl">
              All previous disease evaluations logged locally in SQLite (<code className="text-xs bg-slate-100 px-1 py-0.5 rounded">backend/predictions.db</code>).
            </p>
          </div>

          <button
            onClick={fetchHistory}
            disabled={loading}
            className="inline-flex items-center space-x-2 px-4 py-2.5 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-xl text-xs font-semibold transition cursor-pointer self-start md:self-auto"
          >
            <RotateCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            <span>Refresh Logs</span>
          </button>
        </div>
      </div>

      {/* Filter Bar */}
      <div className="flex items-center space-x-3">
        <div className="relative flex-1 max-w-md">
          <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-3" />
          <input
            type="text"
            placeholder="Filter history by disease or symptom..."
            value={filterQuery}
            onChange={(e) => setFilterQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-white border border-slate-200 rounded-xl text-xs sm:text-sm focus:outline-none focus:ring-2 focus:ring-teal-500 transition"
          />
        </div>
        <span className="text-xs text-slate-500 font-medium">
          Showing {filtered.length} of {historyRecords.length} records
        </span>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-xs text-red-800 flex items-center space-x-2">
          <AlertCircle className="w-4 h-4 text-red-600 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Content */}
      {loading ? (
        <div className="py-24 flex flex-col items-center justify-center space-y-3">
          <Loader2 className="w-8 h-8 animate-spin text-teal-600" />
          <span className="text-xs font-medium text-slate-500">Loading SQLite history records...</span>
        </div>
      ) : filtered.length === 0 ? (
        <div className="bg-white rounded-2xl border border-dashed border-slate-200 p-12 text-center space-y-3">
          <div className="w-12 h-12 rounded-xl bg-slate-100 text-slate-400 flex items-center justify-center mx-auto">
            <HistoryIcon className="w-6 h-6" />
          </div>
          <h3 className="font-bold text-slate-800 text-sm">No Prediction Records Found</h3>
          <p className="text-xs text-slate-500 max-w-sm mx-auto">
            {filterQuery
              ? `No records matching "${filterQuery}". Try clearing the filter.`
              : 'Predictions generated via the Disease Prediction page will automatically be logged here in SQLite.'}
          </p>
        </div>
      ) : (
        <div className="bg-white rounded-2xl border border-slate-200 shadow-xs overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs sm:text-sm border-collapse">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50 text-slate-700 uppercase font-semibold text-2xs tracking-wider">
                  <th className="py-3 px-4">#</th>
                  <th className="py-3 px-4">Date & Time</th>
                  <th className="py-3 px-4">Predicted Disease</th>
                  <th className="py-3 px-4">Selected Symptoms</th>
                  <th className="py-3 px-4">Model Name</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-slate-700">
                {filtered.map((item) => (
                  <tr key={item.id} className="hover:bg-slate-50/70 transition">
                    <td className="py-3 px-4 font-mono text-slate-400 text-xs">
                      #{item.id}
                    </td>
                    <td className="py-3 px-4 whitespace-nowrap text-slate-600">
                      <div className="flex items-center space-x-1.5 text-xs">
                        <Calendar className="w-3.5 h-3.5 text-slate-400" />
                        <span>{item.created_at}</span>
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <span className="font-bold text-teal-900 bg-teal-50 px-2.5 py-1 rounded-lg border border-teal-200 inline-block">
                        {item.predicted_disease}
                      </span>
                    </td>
                    <td className="py-3 px-4 max-w-md">
                      <div className="flex flex-wrap gap-1.5">
                        {Array.isArray(item.symptoms) &&
                          item.symptoms.map((s, idx) => (
                            <span
                              key={idx}
                              className="text-2xs bg-slate-100 text-slate-700 px-2 py-0.5 rounded-md border border-slate-200 capitalize font-medium"
                            >
                              {formatSymptom(s)}
                            </span>
                          ))}
                      </div>
                    </td>
                    <td className="py-3 px-4 whitespace-nowrap">
                      <div className="flex items-center space-x-2">
                        <span className="font-mono text-xs text-slate-700">{item.model_name}</span>
                        <span className={`text-2xs font-bold px-2 py-0.5 rounded-full border ${
                          item.model_version === 'v2.0'
                            ? 'bg-teal-50 text-teal-700 border-teal-200'
                            : 'bg-slate-100 text-slate-500 border-slate-200'
                        }`}>
                          {item.model_version || 'v1.0'}
                        </span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

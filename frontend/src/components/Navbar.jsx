import React from 'react';
import { Activity, Home, Stethoscope, BarChart3, History, CheckCircle2, AlertCircle } from 'lucide-react';

export default function Navbar({ activeTab, setActiveTab, backendStatus }) {
  const navItems = [
    { id: 'home', label: 'Home', icon: Home },
    { id: 'predict', label: 'Disease Prediction', icon: Stethoscope },
    { id: 'models', label: 'Model Comparison', icon: BarChart3 },
    { id: 'history', label: 'History', icon: History },
  ];

  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-50 shadow-xs">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Brand Logo */}
          <div 
            onClick={() => setActiveTab('home')}
            className="flex items-center space-x-3 cursor-pointer group"
          >
            <div className="w-10 h-10 rounded-xl bg-teal-600 flex items-center justify-center text-white shadow-sm group-hover:bg-teal-700 transition">
              <Activity className="w-6 h-6" />
            </div>
            <div>
              <span className="text-lg font-bold text-slate-900 tracking-tight block">
                AI Disease Prediction System
              </span>
              <span className="text-xs text-slate-500 font-medium hidden sm:block">
                Multiclass ML Health Classifier
              </span>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="flex space-x-1 sm:space-x-2">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`flex items-center space-x-2 px-3 sm:px-4 py-2 rounded-lg text-sm font-medium transition cursor-pointer ${
                    isActive
                      ? 'bg-teal-50 text-teal-700 font-semibold shadow-xs'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                  }`}
                >
                  <Icon className={`w-4 h-4 ${isActive ? 'text-teal-600' : 'text-slate-400'}`} />
                  <span>{item.label}</span>
                </button>
              );
            })}
          </nav>

          {/* Status Indicator */}
          <div className="hidden md:flex items-center space-x-2 border-l border-slate-200 pl-4">
            {backendStatus.ready ? (
              <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-50 text-emerald-700 border border-emerald-200">
                <CheckCircle2 className="w-3.5 h-3.5 mr-1 text-emerald-600" />
                API Connected
              </span>
            ) : (
              <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-amber-50 text-amber-700 border border-amber-200">
                <AlertCircle className="w-3.5 h-3.5 mr-1 text-amber-600" />
                Connecting...
              </span>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}

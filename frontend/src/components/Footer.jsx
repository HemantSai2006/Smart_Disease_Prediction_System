import React from 'react';
import { ShieldAlert, Cpu } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="bg-white border-t border-slate-200 mt-auto py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center space-x-2 text-slate-500 text-sm">
            <Cpu className="w-4 h-4 text-teal-600" />
            <span>AI-Based Disease Prediction System • Academic ML Project</span>
          </div>
          <div className="flex items-center space-x-1 text-xs text-amber-700 bg-amber-50 px-3 py-1.5 rounded-md border border-amber-200">
            <ShieldAlert className="w-4 h-4 mr-1 text-amber-600 flex-shrink-0" />
            <span>Not a certified medical diagnostic device. Always seek professional clinical advice.</span>
          </div>
        </div>
      </div>
    </footer>
  );
}

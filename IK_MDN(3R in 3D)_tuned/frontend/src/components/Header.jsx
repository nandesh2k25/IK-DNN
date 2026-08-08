import React from 'react';
import { Cpu } from 'lucide-react';

const Header = () => {
  return (
    <header className="bg-white border-b border-slate-200 h-16 flex items-center px-6 shrink-0 z-10">
      <div className="flex items-center gap-3">
        <div className="bg-blue-600 p-2 rounded-lg">
          <Cpu className="h-6 w-6 text-white" />
        </div>
        <div>
          <h1 className="text-lg font-bold text-slate-900 leading-none">
            MDN-Based Inverse Kinematics System
          </h1>
          <p className="text-xs text-slate-500 mt-1">
            3R Robot in 3D with Multi-Solution Prediction and FK-Based Selection
          </p>
        </div>
      </div>
      
      <div className="ml-auto flex items-center gap-4">
        <div className="h-8 w-px bg-slate-200 mx-2" />
        <div className="flex items-center gap-2">
          <span className="flex h-2 w-2 rounded-full bg-green-500" />
          <span className="text-sm font-medium text-slate-600">System Ready</span>
        </div>
      </div>
    </header>
  );
};

export default Header;

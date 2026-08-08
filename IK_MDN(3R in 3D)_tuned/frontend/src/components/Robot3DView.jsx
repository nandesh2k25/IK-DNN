import React from 'react';
import Plot from 'react-plotly.js';

const Robot3DView = ({ prediction, target, robotParams }) => {
  // Use default params if not provided
  const d1 = robotParams?.d1 || 0.1;
  const a2 = robotParams?.a2 || 0.2;
  const a3 = robotParams?.a3 || 0.2;

  const getRobotTraces = () => {
    if (!prediction) return [];

    const { theta1_rad, theta2_rad, theta3_rad } = prediction.best_component;

    // Joint 1 (Base)
    const x0 = 0, y0 = 0, z0 = 0;
    
    // Joint 2
    const x1 = 0, y1 = 0, z1 = d1;
    
    // Joint 3
    const x2 = a2 * Math.cos(theta1_rad) * Math.cos(theta2_rad);
    const y2 = a2 * Math.sin(theta1_rad) * Math.cos(theta2_rad);
    const z2 = d1 + a2 * Math.sin(theta2_rad);
    
    // End-Effector
    const x3 = (a2 * Math.cos(theta2_rad) + a3 * Math.cos(theta2_rad + theta3_rad)) * Math.cos(theta1_rad);
    const y3 = (a2 * Math.cos(theta2_rad) + a3 * Math.cos(theta2_rad + theta3_rad)) * Math.sin(theta1_rad);
    const z3 = d1 + a2 * Math.sin(theta2_rad) + a3 * Math.sin(theta2_rad + theta3_rad);

    return [
      // Links
      {
        type: 'scatter3d',
        mode: 'lines+markers',
        x: [x0, x1, x2, x3],
        y: [y0, y1, y2, y3],
        z: [z0, z1, z2, z3],
        line: { width: 8, color: '#3b82f6' },
        marker: { size: 6, color: '#1e3a8a' },
        name: 'Robot Arm'
      },
      // Predicted EE
      {
        type: 'scatter3d',
        mode: 'markers',
        x: [x3], y: [y3], z: [z3],
        marker: { size: 8, color: '#10b981', symbol: 'diamond' },
        name: 'Predicted EE'
      }
    ];
  };

  const getTargetTrace = () => {
    return {
      type: 'scatter3d',
      mode: 'markers',
      x: [target.x], y: [target.y], z: [target.z],
      marker: { size: 10, color: '#ef4444', symbol: 'cross' },
      name: 'Target Pose'
    };
  };

  const getErrorLine = () => {
    if (!prediction) return null;
    const { fk_x, fk_y, fk_z } = prediction.best_component;
    return {
      type: 'scatter3d',
      mode: 'lines',
      x: [target.x, fk_x],
      y: [target.y, fk_y],
      z: [target.z, fk_z],
      line: { color: '#f59e0b', width: 4, dash: 'dot' },
      name: 'Error'
    };
  };

  const traces = [
    getTargetTrace(),
    ...getRobotTraces(),
    getErrorLine()
  ].filter(Boolean);

  return (
    <Plot
      data={traces}
      layout={{
        autosize: true,
        margin: { l: 0, r: 0, b: 0, t: 0 },
        scene: {
          xaxis: { range: [-0.6, 0.6], title: 'X' },
          yaxis: { range: [-0.6, 0.6], title: 'Y' },
          zaxis: { range: [0, 0.8], title: 'Z' },
          camera: {
            eye: { x: 1.5, y: 1.5, z: 1.2 }
          },
          aspectmode: 'cube'
        },
        legend: { x: 0, y: 1 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
      }}
      useResizeHandler={true}
      style={{ width: "100%", height: "100%" }}
      config={{ displayModeBar: false }}
    />
  );
};

export default Robot3DView;

import { useState } from 'react';
import FileUpload from "../components/FileUpload";
import Dashboard from "../components/Dashboard";

export default function Home() {
  const [evaluationData, setEvaluationData] = useState(null);

  return (
    <div className="min-h-screen bg-gray-100 p-4">
      <h1 className="text-2xl font-bold text-center mb-6">Welcome to Casca</h1>
      <div className="max-w-2xl mx-auto space-y-6">
        <FileUpload setEvaluationData={setEvaluationData} />
        {evaluationData && <Dashboard evaluationData={evaluationData} />}
      </div>
    </div>
  );
}

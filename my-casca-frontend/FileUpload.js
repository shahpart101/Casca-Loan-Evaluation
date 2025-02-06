import { useState } from 'react';
import axios from 'axios';

const FileUpload = () => {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) return;
  
    const formData = new FormData();
    formData.append('file', file);
  
    try {
      const response = await axios.post('http://localhost:8000/upload/', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResult(response.data);
    } catch (error) {
      console.error('Error uploading file:', error);
    }
  };
  

  return (
    <div className="p-4 bg-white rounded-xl shadow-md">
      <input type="file" onChange={handleFileChange} className="mb-4" />
      <button
        onClick={handleUpload}
        className="px-4 py-2 bg-blue-500 text-white rounded-lg"
      >
        Upload PDF
      </button>

      {result && (
        <div className="mt-4">
          <h2 className="text-xl font-bold">Results:</h2>
          <p>Risk Score: {result.RiskScore}%</p>
          <p>Risk Category: {result.RiskCategory}</p>
          <p>Interest Rate: {result.InterestRate}%</p>
        </div>
      )}
    </div>
  );
};

export default FileUpload;

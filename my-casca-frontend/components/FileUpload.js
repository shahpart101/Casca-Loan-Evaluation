import { useState } from 'react';
import axios from 'axios';

const FileUpload = () => {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    console.log(`📂 Selected file: ${selectedFile.name}`); // Debugging print
  };

  
  const handleUpload = async () => {
    if (!file) {
      console.error("❌ No file selected.");
      return;
    }
  
    const formData = new FormData();
    formData.append('file', file);
  
    try {
      console.log(`📂 Uploading file: ${file.name}`); // Debugging print
  
      const response = await axios.post('http://127.0.0.1:8000/upload/', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
  
      console.log("✅ Upload successful. Response:", response.data);
      setResult(response.data);
    } catch (error) {
      console.error("❌ Error uploading file:", error.response ? error.response.data : error);
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
import { useState, useEffect } from 'react';
import axios from 'axios';

const FileUpload = ({ setEvaluationData }) => {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  // Debug: Log when the component mounts
  useEffect(() => {
    console.log("FileUpload component mounted");
  }, []);

  // Handle file selection
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) {
      console.error("❌ No file selected.");
      setError("Please select a file before uploading.");
      return;
    }
    setFile(selectedFile);
    setError("");
    console.log(`📂 Selected file: ${selectedFile.name}`);
  };

  // Handle file upload
  const handleUpload = async () => {
    console.log("Upload button clicked!");
    if (!file) {
      setError("No file selected! Please choose a file.");
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    setLoading(true);
    setError("");
    setResult(null);

    try {
      console.log(`📂 Uploading file: ${file.name}`);

      // Hardcoded API URL for production (Replace with your live Render backend URL)
      const apiUrl = "https://casca-loan-evaluation.onrender.com/upload/";


      const response = await axios.post(apiUrl, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      console.log("✅ Upload successful. Response:", response.data);
      setResult(response.data);
      // Optionally pass the evaluation data to a parent component
      if (setEvaluationData) {
        setEvaluationData(response.data);
      }
    } catch (error) {
      console.error("❌ Error uploading file:", error.response ? error.response.data : error);
      setError("File upload failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-4 bg-white rounded-xl shadow-md max-w-md mx-auto">
      <h2 className="text-xl font-bold mb-2 text-center">
        Upload a PDF for Loan Evaluation
      </h2>

      <input
        type="file"
        onChange={handleFileChange}
        className="block w-full mb-4 border p-2 rounded-md"
      />

      {error && <p className="text-red-500 text-center">{error}</p>}

      <button
        onClick={handleUpload}
        disabled={loading}
        className="w-full px-4 py-2 mt-4 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50"
      >
        {loading ? "Uploading..." : "Upload PDF"}
      </button>

      {result && (
        <div className="mt-4 p-4 border rounded-md bg-gray-100">
          <h3 className="text-lg font-bold">Results:</h3>
          <p><strong>Risk Score:</strong> {result.RiskScore}%</p>
          <p><strong>Risk Category:</strong> {result.RiskCategory}</p>
          <p><strong>Interest Rate:</strong> {result.InterestRate}%</p>
        </div>
      )}
    </div>
  );
};

export default FileUpload;



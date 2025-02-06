import { useState } from 'react';
import axios from 'axios';

const FileUpload = () => {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];

    if (!selectedFile) {
      console.error("❌ No file selected.");
      setError("Please select a file before uploading.");
      return;
    }

    setFile(selectedFile);
    setError(null);
    console.log(`📂 Selected file: ${selectedFile.name}`);
  };

  const handleUpload = async () => {
    if (!file) {
      setError("No file selected! Please choose a file.");
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      console.log(`📂 Uploading file: ${file.name}`);

      // Use an environment variable for the API URL
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "https://casca-loan-evaluation.onrender.com/upload/";


      const response = await axios.post(apiUrl, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      console.log("✅ Upload successful. Response:", response.data);
      setResult(response.data);
    } catch (error) {
      console.error("❌ Error uploading file:", error.response ? error.response.data : error);
      setError("File upload failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-4 bg-white rounded-xl shadow-md">
      <h2 className="text-xl font-bold mb-2">Upload a PDF for Loan Evaluation</h2>

      <input type="file" onChange={handleFileChange} className="mb-4 border p-2 rounded-md" />

      {error && <p className="text-red-500">{error}</p>}

      <button
        onClick={handleUpload}
        className="px-4 py-2 bg-blue-500 text-white rounded-lg disabled:opacity-50"
        disabled={loading}
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

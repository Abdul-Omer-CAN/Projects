import { useState } from 'react' // import the 'useState' fxn from 'react'.
import './App.css' // import a CSS file for styling

function App() { // defines a JavaScript fxn called App == similiar to python's def App()
  const [age, setAge] = useState('') // 'const' declare a variable that wont be reassigned. [] pulling 2 values out of what useState returns. 'useState('')' react's fxn that pulls an array. useState starts with an empty string ''.
  const [sex, setSex] = useState('') // same as above.
  const [cp, setCp] = useState('')
  const [trestbps, setTrestbps] =  useState('')
  const [chol, setChol] = useState('')
  const [fbs, setFbs] = useState('')
  const [restecg, setRestecg] = useState('')
  const [thalach, setThalach] = useState('')
  const [exang, setExang] = useState('')
  const [oldpeak, setOldpeak] = useState('')
  const [slope, setSlope] = useState('')
  const [ca, setCa] = useState('') 
  const [thal, setThal] = useState('')
  const [result, setResult] = useState(null)

  const handlePredict = async() =>{ // creates a fxn called handlePredict. async= this function will do something that takes time. like waiting for a server response. arrow fxn syntax => opens the fxn body.
    const response = await fetch('http://localhost:8000/predict', { // fetch() Javascript's built in fxn to make HTTP reqs(like calling an API). await means wait until the response of the server comes thru. (URL) of the FastAPI server.
      method: 'POST', // same POST method. Sending data to the server.
      headers: { // more info about the req.
        'Content-Type': 'application/json', // telling server i am sending you JSON formatted data.
      }, // closes the header object.
      body: JSON.stringify({ // body the actual data being sent. JSON.stringify = converts Javascript object to JSON text format(server needs JSON, not raw JS Object.)
        age: parseInt(age), // 'age:' is the key. matches with FastAPI Patientdata. 'parseint' converts the string from our input for e.g "55" into an actual number.
        sex: parseInt(sex),
        cp: parseInt(cp),
        trestbps: parseInt(trestbps),
        chol: parseInt(chol),
        fbs : parseInt(fbs),
        restecg: parseInt(restecg),
        thalach: parseInt(thalach),
        exang: parseInt(exang),
        oldpeak: parseInt(oldpeak),
        slope: parseInt(slope),
        ca: parseInt(ca),
        thal: parseInt(thal)
      })
    })

    const data = await response.json()
    console.log(data)
    setResult(data)
  }

  return( // Here it returns Javascript XML(what gets displayed)
    <div className="App"> {/* this is a wrapper that holds the content below together think of it like a bag and the lines of codes below content. To keep them contained.*/}
      <h1>Clinical Intelligence Platform</h1>  {/*JSX heading element*/}
      <p>Heart Disease Risk Predictor</p> {/*JSX Paragraph element.*/}

      <form> {/* JSX form container element*/}
        <div> {/* Generic container */}
          <label>Age:</label> {/**Text label element */}
          {/* {} curly brackets mean insert Javascript expression here. Sets input's displayed value to our age variable */}
          {/* on change is an event handler runs when input changes. (e) => arrow function & 'e' is the parameter representing the event. 'e.target.value' is accessing the input's current typed value. setAge() calling our update fxn with the new value.*/}
          <input
          type="number"
          value={age} 
          onChange={(e) => setAge(e.target.value)} 
          />
        </div>

        <div>
          <label>Sex:</label>
          <select value={sex} onChange={(e) => setSex(e.target.value)}>
            <option value="">Select</option>
            <option value="1">Male</option>
            <option value="0">Female</option>
          </select>
        </div>

        <div>
          <label>Chest Pain Type:</label>
          <select value={cp} onChange={(e) => setCp(e.target.value)}>
            <option value="">Select</option>
            <option value="0">Typical Angina</option>
            <option value="1">Atypical Angina</option>
            <option value="2">Non-anginal Pain</option>
            <option value="3">Asymtomatic</option>
          </select>
        </div>

        <div>
          <label>Resting Blood Pressure:</label>
          <input
          type="number"
          value={trestbps}
          onChange={(e) => setTrestbps(e.target.value)}
          />
        </div>

        <div>
          <label>Cholesterol:</label>
          <input
          type="number"
          value={chol}
          onChange={(e) => setChol(e.target.value)}
          />
        </div>

        <div>
          <label>Fasting Blood Sugar {'>'}120 mg/dl:</label>
          {/* onchange runs when dropdown selection is selected. (e) => arrow fxn, 'e' is the event parameter. 
          setFbs() updates our fbs state. e.target the select element that triggered the event. 
          e.target.value the currently selected option's value. */}
          <select value={fbs} onChange={(e) => setFbs(e.target.value)}>
            <option value="">Select</option>
            <option value="1">Yes</option>
            <option value="2">No</option>
          </select>
        </div>

        <div>
          <label>Resting ECG Results:</label>
          <select value={restecg} onChange={(e) => setRestecg(e.target.value)}>
            <option value="">Select</option>
            <option value="0">Normal</option>
            <option value="1">ST-T Wave Abnormality</option>
            <option value="2">Left Ventricular Hypertrophy</option>
          </select>
        </div>

        <div>
          <label>Maximum Heart Rate Achieved:</label>
          <input
          type="number"
          value={thalach}
          onChange={(e) => setThalach(e.target.value)}
          />
        </div>

        <div>
          <label>Exercise Induced Angina:</label>
          <select value={exang} onChange={(e) => setExang(e.target.value)}>
            <option value="">Select</option>
            <option value="1">Yes</option>
            <option value="2">No</option>
          </select>
        </div>

        <div>
          <label>ST Depression (Oldpeak):</label>
          <input
           type="number"
           step="0.1"
           value={oldpeak}
           onChange={(e) => setOldpeak(e.target.value)}
           />
        </div>

        <div>
          <label>Slope of Peak Exercise ST Segment:</label>
          <select value ={slope} onChange={(e) => setSlope(e.target.value)}>
            <option value="">Select</option>
            <option value="0">Upsloping</option>
            <option value="1">Flat</option>
            <option value="2">Downsloping</option>
          </select>
        </div>

        <div>
          <label>Number of Major Vessels (0-4):</label>
          <select value={ca} onChange={(e) => setCa(e.target.value)}>
            <option value="">Select</option>
            <option value="0">0</option>
            <option value="1">1</option>
            <option value="2">2</option>
            <option value="3">3</option>
            <option value="4">4</option>
          </select>
        </div>

        <div>
          <label>Thalassemia:</label>
          <select value={thal} onChange={(e) => setThal(e.target.value)}>
          <option value="">Select</option>
          <option value="1">Normal</option>
          <option value="2">Fixed Defect</option>
          <option value="3">Reversible Defect</option>
          </select>
        </div>

        <button type="button" onClick={handlePredict}>Predict</button>

      </form>

      {result && (
        <div>
          <h2>Prediction Result:</h2>
          <p>{result.message}</p>
          <p>Risk: {result.risk}</p>
        </div>
      )}
    </div>
  )
}

export default App


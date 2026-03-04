// pages/_app.jsx
import Chatbot from '../components/Chatbot';
import '../styles/globals.css';

function MyApp({ Component, pageProps }) {
  return (
    <>
      <Component {...pageProps} />
      <Chatbot 
        botName="Medical Chatbot"
        primaryColor="#10b981"
        botAvatar="https://cdn-icons-png.flaticon.com/512/387/387569.png"
        userAvatar="https://i.ibb.co/d5b84Xw/Untitled-design.png"
        apiEndpoint="/get"
        placeholder="Écrivez votre message..."
      />
    </>
  );
}

export default MyApp;
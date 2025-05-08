import React, { useContext } from 'react'
import { assets } from '../assets/assets'
import { useNavigate } from 'react-router-dom'
import { AppContext } from '../context/AppContext'

const Banner = () => {
  const { token } = useContext(AppContext)
  const navigate = useNavigate()

  if (token) return null // Don't render anything if logged in

  return (
    <div className="flex flex-col md:flex-row items-center bg-primary rounded-2xl px-6 sm:px-10 md:px-14 lg:px-20 py-12 my-20 md:mx-10">
      {/* Left Side */}
      <div className="flex-1 text-center md:text-left">
        <div className="text-2xl sm:text-3xl md:text-4xl lg:text-5xl font-semibold text-white">
          <p>Schedule Your Appointment</p>
          <p className="mt-3">with Trusted Medical Experts</p>
        </div>
        <button
          onClick={() => {
            navigate('/login')
            scrollTo(0, 0)
          }}
          className="bg-white text-sm sm:text-base text-[#595959] px-8 py-3 rounded-full mt-6 hover:scale-105 transition-all"
        >
          Create account
        </button>
      </div>

      {/* Right Side */}
      <div className="flex justify-center mt-10 md:mt-0 md:ml-8 w-full md:w-1/2 lg:w-[600px]">
        <img
          src={assets.appointment_img}
          alt="Doctors"
          className="w-full max-w-[340px] object-contain"
        />
      </div>
    </div>
  )
}

export default Banner

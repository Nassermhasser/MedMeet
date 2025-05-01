import React from 'react'
import { assets } from '../assets/assets'
import { useNavigate } from 'react-router-dom'

const Banner = () => {
  const navigate = useNavigate()

  return (
    <div className='flex flex-col md:flex-row items-center bg-teal-500 text-white rounded-3xl px-6 md:px-14 lg:px-20 py-12 my-20 shadow-md'>
      {/* Left */}
      <div className='flex-1 mb-6 md:mb-0'>
        <div className='text-3xl sm:text-4xl lg:text-5xl font-semibold'>
          <p>Book Your Appointment</p>
          <p className='mt-3'>with Morocco’s Most Trusted Doctors</p>
        </div>
        <button
          onClick={() => { navigate('/login'); scrollTo(0, 0) }}
          className='mt-6 bg-white text-teal-700 px-8 py-3 rounded-full font-medium shadow hover:bg-gray-100 transition'
        >
          Create account
        </button>
      </div>

      {/* Right */}
      <div className='md:w-1/2 lg:w-[400px] relative'>
        <img className='w-full md:absolute bottom-0 max-w-md drop-shadow-xl' src={assets.appointment_img} alt="Appointment" />
      </div>
    </div>
  )
}

export default Banner

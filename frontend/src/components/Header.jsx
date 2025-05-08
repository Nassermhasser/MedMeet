import React from 'react'
import { assets } from '../assets/assets'

const Header = () => {
  return (
    <div className="flex flex-col md:flex-row bg-primary rounded-lg px-6 md:px-10 lg:px-20 py-12 md:py-20 items-center md:items-start gap-10">
      {/* --------- Header Left --------- */}
      <div className="md:w-1/2 flex flex-col gap-5 text-white text-start">
        <p className="text-3xl md:text-4xl lg:text-5xl font-semibold leading-tight">
          Schedule Your Appointment <br /> with Trusted Medical Experts
        </p>
        <div className="flex flex-col md:flex-row items-center gap-3 text-sm font-light">
          <img className="w-28" src={assets.group_profiles} alt="Doctors group" />
          <p className="text-center md:text-left">
            Easily explore our wide network of verified doctors,
            <br className="hidden sm:block" />
            and book your appointment with confidence, without any hassle.
          </p>
        </div>
        <a
          href="#speciality"
          className="flex items-center gap-2 bg-white px-8 py-3 rounded-full text-[#595959] text-sm hover:scale-105 transition-all duration-300 w-fit"
        >
          Book appointment
          <img className="w-3" src={assets.arrow_icon} alt="Arrow" />
        </a>
      </div>

      {/* --------- Header Right --------- */}
      <div className="md:w-1/2 flex justify-center">
        <img
          className="w-full max-w-[450px] h-auto rounded-xl object-contain"
          src={assets.header_img}
          alt="Header Visual"
        />
      </div>
    </div>
  )
}

export default Header

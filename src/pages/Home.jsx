import React from 'react'
import Header from '../components/Header'
import SpecialityMenu from '../components/SpecialityMenu'
import TopDoctors from '../components/TopDoctors'
import Banner from '../components/Banner'

const Home = () => {
  return (
    <div className='bg-[#f9fafe] text-[#262626]'>
      <Header />
      <div className='px-4 sm:px-8 md:px-12 lg:px-20'>
        <SpecialityMenu />
        <TopDoctors />
      </div>
      <Banner />
    </div>
  )
}

export default Home

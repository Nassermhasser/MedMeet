import React, { useContext, useState } from 'react'
import { assets } from '../assets/assets'
import { NavLink, useNavigate } from 'react-router-dom'
import { AppContext } from '../context/AppContext'

const Navbar = () => {
  const navigate = useNavigate()
  const [showMenu, setShowMenu] = useState(false)
  const { token, setToken, userData } = useContext(AppContext)

  const logout = () => {
    localStorage.removeItem('token')
    setToken(false)
    navigate('/login')
  }

  return (
    <div className='flex items-center justify-between px-6 py-4 bg-white shadow-sm border-b border-gray-200'>
      <img onClick={() => navigate('/')} className='w-44 cursor-pointer' src={assets.logo} alt="Logo" />

      <ul className='md:flex items-center gap-6 font-medium hidden text-gray-600'>
        <NavLink to='/'><li className='hover:text-teal-600 transition'>HOME</li></NavLink>
        <NavLink to='/doctors'><li className='hover:text-teal-600 transition'>DOCTORS</li></NavLink>
        <NavLink to='/about'><li className='hover:text-teal-600 transition'>ABOUT</li></NavLink>
        <NavLink to='/contact'><li className='hover:text-teal-600 transition'>CONTACT</li></NavLink>
      </ul>

      <div className='flex items-center gap-4'>
        {token && userData ? (
          <div className='relative group'>
            <img className='w-8 rounded-full cursor-pointer' src={userData.image} alt="User" />
            <img className='w-2.5 ml-1' src={assets.dropdown_icon} alt="Dropdown" />
            <div className='absolute right-0 pt-12 hidden group-hover:block z-20'>
              <div className='min-w-48 bg-white shadow-md rounded-xl p-4 text-sm text-gray-700 space-y-2'>
                <p onClick={() => navigate('/my-profile')} className='hover:text-black cursor-pointer'>My Profile</p>
                <p onClick={() => navigate('/my-appointments')} className='hover:text-black cursor-pointer'>My Appointments</p>
                <p onClick={logout} className='hover:text-black cursor-pointer'>Logout</p>
              </div>
            </div>
          </div>
        ) : (
          <button
            onClick={() => navigate('/login')}
            className='bg-teal-500 hover:bg-teal-600 text-white px-6 py-2.5 rounded-full font-medium shadow transition hidden md:block'
          >
            Create account
          </button>
        )}

        <img onClick={() => setShowMenu(true)} className='w-6 md:hidden' src={assets.menu_icon} alt="Menu" />
      </div>

      {/* Mobile Menu */}
      <div className={`md:hidden ${showMenu ? 'fixed w-full' : 'h-0 w-0'} right-0 top-0 bottom-0 z-20 overflow-hidden bg-white transition-all`}>
        <div className='flex items-center justify-between px-5 py-6'>
          <img src={assets.logo} className='w-36' alt="Logo" />
          <img onClick={() => setShowMenu(false)} src={assets.cross_icon} className='w-7' alt="Close" />
        </div>
        <ul className='flex flex-col items-center gap-4 mt-5 px-5 text-base font-semibold text-gray-700'>
          <NavLink onClick={() => setShowMenu(false)} to='/'><p className='px-4 py-2 hover:bg-teal-100 w-full text-center rounded-md'>HOME</p></NavLink>
          <NavLink onClick={() => setShowMenu(false)} to='/doctors'><p className='px-4 py-2 hover:bg-teal-100 w-full text-center rounded-md'>DOCTORS</p></NavLink>
          <NavLink onClick={() => setShowMenu(false)} to='/about'><p className='px-4 py-2 hover:bg-teal-100 w-full text-center rounded-md'>ABOUT</p></NavLink>
          <NavLink onClick={() => setShowMenu(false)} to='/contact'><p className='px-4 py-2 hover:bg-teal-100 w-full text-center rounded-md'>CONTACT</p></NavLink>
        </ul>
      </div>
    </div>
  )
}

export default Navbar

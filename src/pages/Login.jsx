import React, { useContext, useEffect, useState } from 'react'
import { AppContext } from '../context/AppContext'
import axios from 'axios'
import { toast } from 'react-toastify'
import { useNavigate } from 'react-router-dom'

const Login = () => {
  const [state, setState] = useState('Sign Up')
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const navigate = useNavigate()
  const { backendUrl, token, setToken } = useContext(AppContext)

  const onSubmitHandler = async (event) => {
    event.preventDefault()

    try {
      if (state === 'Sign Up') {
        const { data } = await axios.post(backendUrl + '/api/user/register', { name, email, password })
        if (data.success) {
          localStorage.setItem('token', data.token)
          setToken(data.token)
        } else {
          toast.error(data.message)
        }
      } else {
        const { data } = await axios.post(backendUrl + '/api/user/login', { email, password })
        if (data.success) {
          localStorage.setItem('token', data.token)
          setToken(data.token)
        } else {
          toast.error(data.message)
        }
      }
    } catch (error) {
      toast.error("An error occurred. Please try again.")
    }
  }

  useEffect(() => {
    if (token) {
      navigate('/')
    }
  }, [token])

  return (
    <form onSubmit={onSubmitHandler} className='min-h-[80vh] flex items-center justify-center px-4'>
      <div className='w-full max-w-md bg-white border border-gray-200 rounded-2xl shadow-lg p-8 space-y-4 text-[#4A4A4A]'>
        <div className='text-center'>
          <h2 className='text-2xl font-bold text-[#222]'>{state === 'Sign Up' ? 'Create Account' : 'Login'}</h2>
          <p className='text-sm text-gray-500'>Please {state === 'Sign Up' ? 'sign up' : 'log in'} to book an appointment</p>
        </div>

        {state === 'Sign Up' && (
          <div>
            <label className='text-sm'>Full Name</label>
            <input
              type='text'
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
              className='w-full mt-1 p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-200'
            />
          </div>
        )}

        <div>
          <label className='text-sm'>Email</label>
          <input
            type='email'
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            className='w-full mt-1 p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-200'
          />
        </div>

        <div>
          <label className='text-sm'>Password</label>
          <input
            type='password'
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            className='w-full mt-1 p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-200'
          />
        </div>

        <button className='w-full bg-primary hover:bg-blue-700 text-white py-2 rounded-md font-medium transition'>
          {state === 'Sign Up' ? 'Create account' : 'Login'}
        </button>

        <div className='text-center text-sm'>
          {state === 'Sign Up' ? (
            <p>Already a member? <span onClick={() => setState('Login')} className='text-primary underline cursor-pointer'>Login here</span></p>
          ) : (
            <p>Create an account? <span onClick={() => setState('Sign Up')} className='text-primary underline cursor-pointer'>Click here</span></p>
          )}
        </div>
      </div>
    </form>
  )
}

export default Login

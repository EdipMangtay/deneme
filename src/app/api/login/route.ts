// Next Imports
import { NextResponse } from 'next/server'

import type { UserTable } from './users'

type ResponseUser = Omit<UserTable, 'password'>

import { users } from './users'

export async function POST(req: Request) {
  const { email, password } = await req.json()
  const user = users.find(u => u.email === email && u.password === password)
  let response: null | ResponseUser = null

  if (user) {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const { password: _, ...filteredUserData } = user

    response = {
      ...filteredUserData
    }

    return NextResponse.json(response)
  } else {
    return NextResponse.json(
      {
        message: ['Email or Password is invalid']
      },
      {
        status: 401,
        statusText: 'Unauthorized Access'
      }
    )
  }
}
